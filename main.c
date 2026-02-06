#include <stdio.h>
#include <math.h>
#include <string.h>

#include "pico/stdlib.h"
#include "pico/time.h"
#include "pico/multicore.h"
#include "pico/mutex.h"

#include "hardware/pwm.h"
#include "hardware/i2c.h"
#include "hardware/gpio.h"
#include "hardware/watchdog.h"
#include "hardware/structs/watchdog.h"

#ifdef __cplusplus
extern "C" {
#endif

#include "lib/ssd1306.h"
#include "lib/font.h"
#include "lib/aht20.h"

#ifdef __cplusplus
}
#endif

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "autoencoder_temperatura_float.h"

#define I2C_PORT_A i2c0
#define I2C_SDA_A 0
#define I2C_SCL_A 1

#define BOT_A 5
#define BOT_B 6

#define I2C_PORT_B i2c1
#define I2C_SDA_B 14
#define I2C_SCL_B 15

#define I2C_ENDERECO 0x3c
#define BUZZER_A 21

ssd1306_t ssd;
AHT20_Data data;

int temperature = 0; 
float humidity = 0;
float media = 0;
volatile int temp_offset_centi = 0;

char str_tmp[10];
char str_umi[10];
char str_mse[16];
char str_sta[16];

typedef struct checks {
    bool error1;
    bool error2;
    bool error3;
} checks;
checks c = {false, false, false};

typedef struct WD_checks {
    uint16_t error001;
    uint16_t error002;
    uint16_t error003;
    uint16_t error004;
} WD_checks;
WD_checks wdc = {0, 0, 0, 0};

typedef struct live_checks {
    uint32_t core_0;
    uint32_t core_1;
} live_checks;
live_checks alive = {0, 0};

uint8_t slice = 0;

typedef struct {
    float dc;           /* wrap value do PWM */
    float div;          /* divisor de clock */
    bool alarm_state;   /* estado da sirene */
    bool alarm_react;   /* permite reativar alarme */
    alarm_id_t alarm_pwm;
} pwm_struct;
pwm_struct pw = {7812.5f, 32.0f, false, true, 0};

mutex_t data_mutex;

/* ----------------- ML params ----------------- */
#define ML_WINDOW 20

static const float kTempMin[ML_WINDOW] = {
    23.518806f, 23.501051f, 23.398729f, 23.345318f,
    23.482470f, 23.350306f, 23.482172f, 23.507483f,
    23.464437f, 23.338785f, 23.399324f, 23.281837f,
    23.465727f, 23.429327f, 23.211569f, 23.501291f,
    23.508408f, 23.430216f, 23.373257f, 23.394426f
};

static const float kTempMax[ML_WINDOW] = {
    25.340790f, 25.398533f, 25.382153f, 25.563015f,
    25.485449f, 25.382772f, 25.364466f, 25.525917f,
    25.507328f, 25.608130f, 25.403925f, 25.622328f,
    25.527430f, 25.765596f, 25.698732f, 25.681974f,
    25.662485f, 25.911837f, 25.718164f, 25.771938f
};

static const float kAnomalyThreshold = 0.030335f;

/* Ring buffer */
static float temp_window[ML_WINDOW];
static int win_count = 0;
static int win_head = 0;

/* Estado ML */
static volatile bool ml_ready = false;
static volatile bool ml_anomaly = false;
static volatile float ml_last_mse = 0.0f;

/* TFLM runtime */
static const tflite::Model *model = NULL;
static tflite::MicroInterpreter *interpreter = NULL;
static TfLiteTensor *input = NULL;
static TfLiteTensor *output = NULL;

static constexpr int kTensorArenaSize = 20 * 1024;
static uint8_t tensor_arena[kTensorArenaSize];

/* --------------- Prototypes --------------- */
void core1(void);
void init_botoes(void);
void init_i2c0(void);
void init_i2c1(void);
void init_oled(void);
void init_aht20(void);

void pwm_setup(void);
void pwm_on(uint8_t duty_cycle);
void pwm_off(void);

void gpio_irq_handler(uint gpio, uint32_t events);
int64_t variacao_temp(alarm_id_t, void *user_data);
void wd_errors(uint16_t x, bool positive);

static inline float clamp01(float x);
static inline float minmax_scale(float x, int i);
static void push_temp(float t);
static void get_window_ordered(float out[ML_WINDOW]);

static void ml_init(void);
static bool ml_is_anomaly(const float window[ML_WINDOW], float *out_mse);

int main() {
    wd_errors(3, 1);
    stdio_init_all();

    if (watchdog_caused_reboot()) {
        sleep_ms(3000);
        uint16_t reset_count = watchdog_hw->scratch[0];
        reset_count++;
        watchdog_hw->scratch[0] = reset_count;

        printf("\n\n>>> Reiniciado pelo Watchdog! Contagem de resets: %d\n", reset_count);
        printf("Erro001 - %d\n", wdc.error001);
        printf("Erro002 - %d\n", wdc.error002);
        printf("Erro003 - %d\n", wdc.error003);
        printf("Erro004 - %d\n", wdc.error004);
    } else {
        printf(">>> Reset normal (Power On). Iniciando contador em 0.\n");
        watchdog_hw->scratch[0] = 0;
        watchdog_hw->scratch[1] = 0;
        watchdog_hw->scratch[2] = 0;
        watchdog_hw->scratch[3] = 0;
        watchdog_hw->scratch[4] = 0;
    }

    watchdog_enable(8000, true);

    mutex_init(&data_mutex);

    multicore_launch_core1(core1);

    init_botoes();
    init_i2c0();
    init_aht20();
    pwm_setup();

    ml_init();

    gpio_set_irq_enabled_with_callback(BOT_A, GPIO_IRQ_EDGE_FALL, true, &gpio_irq_handler);
    gpio_set_irq_enabled_with_callback(BOT_B, GPIO_IRQ_EDGE_FALL, true, &gpio_irq_handler);

    wd_errors(3, 0);

    uint32_t last_sample_ms = 0;

    while (true) {
        wd_errors(1, 1);

        alive.core_0 = to_ms_since_boot(get_absolute_time());

        uint32_t now_ms = alive.core_0;

        if ((now_ms - last_sample_ms) >= 1000) {
            last_sample_ms = now_ms;

            if (aht20_read(I2C_PORT_A, &data)) {
                c.error3 = true;

                mutex_enter_blocking(&data_mutex);

                float temp_display = data.temperature + (temp_offset_centi / 100.0f);
                temperature = (int)lroundf(temp_display * 100.0f);
                humidity = data.humidity;

                snprintf(str_tmp, sizeof(str_tmp), "%.1fC", temp_display);
                snprintf(str_umi, sizeof(str_umi), "%.1f%%", humidity);

                mutex_exit(&data_mutex);

                push_temp(temp_display);

                if (win_count >= ML_WINDOW) {
                    float w[ML_WINDOW];
                    get_window_ordered(w);

                    float mse = 0.0f;
                    bool anom = ml_is_anomaly(w, &mse);

                    ml_last_mse = mse;
                    ml_anomaly = anom;

                    mutex_enter_blocking(&data_mutex);
                    snprintf(str_mse, sizeof(str_mse), "MSE:%0.3f", mse);
                    snprintf(str_sta, sizeof(str_sta), "%s", anom ? "ANOMALIA" : "NORMAL");
                    mutex_exit(&data_mutex);

                    if (anom && !pw.alarm_state && pw.alarm_react) {
                        pw.alarm_pwm = add_alarm_in_ms(3000, variacao_temp, NULL, false);
                        pw.alarm_state = true;
                        pw.alarm_react = false;
                    }

                    if (!anom) {
                        if (pw.alarm_state) {
                            pwm_off();
                            pw.alarm_state = false;
                            cancel_alarm(pw.alarm_pwm);
                        }
                        pw.alarm_react = true;
                    }
                }

                printf("Temp: %.2f C | Umi: %.2f %% | MSE: %.5f | %s\n",
                       data.temperature, data.humidity, (double)ml_last_mse,
                       ml_anomaly ? "ANOM" : "OK");
            } else {
                c.error3 = false;
                printf("Erro na leitura do AHT20!\n");
            }

            media = ((temperature / 100.0f) + data.temperature) / 2.0f;
        }

        if (!c.error2 && !c.error3) {
            c.error1 = true;
        } else {
            c.error1 = false;
        }

        int32_t diff = (int32_t)alive.core_0 - (int32_t)alive.core_1;
        if (diff < 0) {
            diff = -diff;
        }
        if (diff < 4000) {
            watchdog_update();
        }

        sleep_ms(10);
        wd_errors(1, 0);
    }
}

void core1(void) {
    wd_errors(4, 1);
    sleep_ms(2000);

    bool cor = true;
    init_i2c1();
    init_oled();

    mutex_enter_blocking(&data_mutex);
    snprintf(str_mse, sizeof(str_mse), "MSE:---");
    snprintf(str_sta, sizeof(str_sta), "INICIANDO");
    mutex_exit(&data_mutex);

    wd_errors(4, 0);

    while (true) {
        wd_errors(2, 1);

        alive.core_1 = to_ms_since_boot(get_absolute_time());

        ssd1306_fill(&ssd, !cor);
        ssd1306_rect(&ssd, 3, 3, 122, 60, cor, !cor);
        ssd1306_line(&ssd, 3, 38, 123, 38, cor);
        ssd1306_line(&ssd, 3, 50, 123, 50, cor);
        ssd1306_line(&ssd, 3, 15, 123, 15, cor);
        ssd1306_line(&ssd, 3, 27, 123, 27, cor);
        ssd1306_draw_string(&ssd, " UMI    TEMP", 10, 41);
        ssd1306_line(&ssd, 63, 39, 63, 60, cor);

        mutex_enter_blocking(&data_mutex);
        if (c.error3) {
            ssd1306_draw_string(&ssd, str_tmp, 73, 52);
            ssd1306_draw_string(&ssd, str_umi, 13, 52);
        } else {
            ssd1306_draw_string(&ssd, "---", 73, 52);
            ssd1306_draw_string(&ssd, "---", 13, 52);
        }

        ssd1306_draw_string(&ssd, str_sta, 10, 6);
        ssd1306_draw_string(&ssd, str_mse, 10, 18);
        mutex_exit(&data_mutex);

        ssd1306_send_data(&ssd);

        wd_errors(2, 0);
        sleep_ms(250);
    }
}

void init_botoes(void) {
    for (uint8_t botoes = 5; botoes < 7; botoes++) {
        gpio_init(botoes);
        gpio_set_dir(botoes, GPIO_IN);
        gpio_pull_up(botoes);
    }
}

void init_i2c0(void) {
    i2c_init(I2C_PORT_A, 400 * 1000);
    for (uint8_t pin = 0; pin < 2; pin++) {
        gpio_set_function(pin, GPIO_FUNC_I2C);
        gpio_pull_up(pin);
    }
}

void init_i2c1(void) {
    i2c_init(I2C_PORT_B, 400 * 1000);
    for (uint8_t pin = 14; pin < 16; pin++) {
        gpio_set_function(pin, GPIO_FUNC_I2C);
        gpio_pull_up(pin);
    }
}

void init_oled(void) {
    ssd1306_init(&ssd, WIDTH, HEIGHT, false, I2C_ENDERECO, I2C_PORT_B);
    ssd1306_config(&ssd);
    ssd1306_fill(&ssd, false);
    ssd1306_send_data(&ssd);
}

void init_aht20(void) {
    aht20_reset(I2C_PORT_A);
    aht20_init(I2C_PORT_A);
}

void pwm_setup(void) {
    gpio_set_function(BUZZER_A, GPIO_FUNC_PWM);
    slice = pwm_gpio_to_slice_num(BUZZER_A);
    pwm_set_clkdiv(slice, pw.div);
    pwm_set_wrap(slice, pw.dc);
    pwm_set_enabled(slice, false);
}

void pwm_on(uint8_t duty_cycle) {
    gpio_set_function(BUZZER_A, GPIO_FUNC_PWM);
    pwm_set_gpio_level(BUZZER_A, (uint16_t)((pw.dc * duty_cycle) / 100));
    pwm_set_enabled(slice, true);
}

void pwm_off(void) {
    pwm_set_enabled(slice, false);
    gpio_set_function(BUZZER_A, GPIO_FUNC_SIO);
    gpio_put(BUZZER_A, 0);
}

void gpio_irq_handler(uint gpio, uint32_t events) {
    (void)events;

    uint64_t current_time = to_ms_since_boot(get_absolute_time());
    static uint64_t last_time_a = 0, last_time_b = 0;

    if (gpio == BOT_A && (current_time - last_time_a > 300)) {
        pwm_off();
        pw.alarm_state = false;
        pw.alarm_react = false;
        cancel_alarm(pw.alarm_pwm);
        last_time_a = current_time;
    }

    if (gpio == BOT_B && (current_time - last_time_b > 300)) {
        temp_offset_centi = (temp_offset_centi == 0) ? 3500 : 0;
        last_time_b = current_time;
    }
}

int64_t variacao_temp(alarm_id_t, void *user_data) {
    (void)user_data;
    pwm_on(50);
    return 1;
}

void wd_errors(uint16_t x, bool positive) {
    switch (x) {
        case 1:
            if (positive) {
                wdc.error001 = watchdog_hw->scratch[1];
                wdc.error001++;
                watchdog_hw->scratch[1] = wdc.error001;
            } else {
                wdc.error001 = watchdog_hw->scratch[1];
                wdc.error001--;
                watchdog_hw->scratch[1] = wdc.error001;
            }
            break;

        case 2:
            if (positive) {
                wdc.error002 = watchdog_hw->scratch[2];
                wdc.error002++;
                watchdog_hw->scratch[2] = wdc.error002;
            } else {
                wdc.error002 = watchdog_hw->scratch[2];
                wdc.error002--;
                watchdog_hw->scratch[2] = wdc.error002;
            }
            break;

        case 3:
            if (positive) {
                wdc.error003 = watchdog_hw->scratch[3];
                wdc.error003++;
                watchdog_hw->scratch[3] = wdc.error003;
            } else {
                wdc.error003 = watchdog_hw->scratch[3];
                wdc.error003--;
                watchdog_hw->scratch[3] = wdc.error003;
            }
            break;

        case 4:
            if (positive) {
                wdc.error004 = watchdog_hw->scratch[4];
                wdc.error004++;
                watchdog_hw->scratch[4] = wdc.error004;
            } else {
                wdc.error004 = watchdog_hw->scratch[4];
                wdc.error004--;
                watchdog_hw->scratch[4] = wdc.error004;
            }
            break;
    }
}

static inline float clamp01(float x) {
    if (x < 0.0f) {
        return 0.0f;
    }
    if (x > 1.0f) {
        return 1.0f;
    }
    return x;
}

static inline float minmax_scale(float x, int i) {
    float den = kTempMax[i] - kTempMin[i];
    if (den == 0.0f) {
        den = 1.0f;
    }
    return clamp01((x - kTempMin[i]) / den);
}

static void push_temp(float t) {
    temp_window[win_head] = t;
    win_head = (win_head + 1) % ML_WINDOW;
    if (win_count < ML_WINDOW) {
        win_count++;
    }
}

static void get_window_ordered(float out[ML_WINDOW]) {
    int start = (win_head - win_count + ML_WINDOW) % ML_WINDOW;
    for (int i = 0; i < win_count; i++) {
        out[i] = temp_window[(start + i) % ML_WINDOW];
    }
    for (int i = win_count; i < ML_WINDOW; i++) {
        out[i] = out[win_count - 1];
    }
}

static void ml_init(void) {
    model = tflite::GetModel(autoencoder_temperatura_float_tflite);
    if (model == NULL) {
        printf("❌ GetModel retornou NULL\n");
        while (1) {
            tight_loop_contents();
        }
    }

    static tflite::MicroMutableOpResolver<4> resolver;
    resolver.AddFullyConnected();
    resolver.AddRelu();
    resolver.AddReshape();

    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize
    );
    interpreter = &static_interpreter;

    TfLiteStatus status = interpreter->AllocateTensors();
    if (status != kTfLiteOk) {
        printf("❌ AllocateTensors falhou\n");
        while (1) {
            tight_loop_contents();
        }
    }

    input = interpreter->input(0);
    output = interpreter->output(0);

    ml_ready = true;
    printf("✅ ML inicializado\n");
}

static bool ml_is_anomaly(const float window[ML_WINDOW], float *out_mse) {
    if (!ml_ready || interpreter == NULL || input == NULL || output == NULL) {
        return false;
    }

    for (int i = 0; i < ML_WINDOW; i++) {
        input->data.f[i] = minmax_scale(window[i], i);
    }

    if (interpreter->Invoke() != kTfLiteOk) {
        return false;
    }

    float mse = 0.0f;
    for (int i = 0; i < ML_WINDOW; i++) {
        float diff = output->data.f[i] - input->data.f[i];
        mse += diff * diff;
    }
    mse /= (float)ML_WINDOW;

    if (out_mse != NULL) {
        *out_mse = mse;
    }
    return (mse > kAnomalyThreshold);
}