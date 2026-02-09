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
char str_sta[24];

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
#define ML_WINDOW 60
#define COND_STABLE_SECONDS 10
#define COND_STABLE_DELTA 0.5f
#define COND_ELEVADA_DELTA 2.0f
#define COND_SEVERA_DELTA 5.0f
#define COND_SEVERE_WINDOW_MS 60000

static const float kTempMin[ML_WINDOW] = {
    24.000195f, 23.800744f, 23.753186f, 23.651148f, 23.478117f,
    23.543856f, 23.423320f, 23.454261f, 23.414481f, 23.445453f,
    23.336992f, 23.253057f, 23.183236f, 23.202494f, 23.139543f,
    23.078896f, 23.082267f, 23.049465f, 22.967841f, 22.862567f,
    22.753356f, 22.851117f, 22.832863f, 22.758146f, 22.657734f,
    22.523301f, 22.451465f, 22.307808f, 22.264346f, 22.193564f,
    22.115642f, 22.239464f, 22.215062f, 22.196853f, 22.325051f,
    22.189177f, 22.140474f, 21.985550f, 21.992578f, 22.053292f,
    21.853292f, 21.939291f, 21.970395f, 21.974044f, 21.849859f,
    21.927053f, 22.095343f, 22.054594f, 22.022848f, 22.025123f,
    21.985177f, 21.967361f, 22.070211f, 22.063579f, 22.092584f,
    22.292584f, 22.320498f, 22.287693f, 22.242896f, 22.101548f
};

static const float kTempMax[ML_WINDOW] = {
    24.999672f, 25.163015f, 25.359282f, 25.421523f, 25.478898f,
    25.443300f, 25.435772f, 25.566029f, 25.600808f, 25.565483f,
    25.584186f, 25.559955f, 25.615816f, 25.732980f, 25.729611f,
    25.721083f, 25.785393f, 25.901790f, 25.958078f, 25.870097f,
    26.068803f, 26.015706f, 25.982192f, 26.052790f, 26.171842f,
    26.257239f, 26.298726f, 26.345055f, 26.349406f, 26.405220f,
    26.205220f, 26.292078f, 26.337232f, 26.368740f, 26.357379f,
    26.367425f, 26.482223f, 26.650305f, 26.802097f, 26.794899f,
    26.885127f, 26.855321f, 26.849877f, 26.920392f, 26.862736f,
    26.910096f, 26.847646f, 26.794115f, 26.873489f, 26.783930f,
    26.677477f, 26.733249f, 26.684430f, 26.656430f, 26.581355f,
    26.592485f, 26.656157f, 26.770851f, 26.763900f, 26.878825f
};

static const float kAnomalyThreshold = 0.007699098f;

/* Ring buffer */
static float temp_window[ML_WINDOW];
static int win_count = 0;
static int win_head = 0;

/* Estado ML */
static volatile bool ml_ready = false;
static volatile bool ml_anomaly = false;
static volatile float ml_last_mse = 0.0f;

/* Baseline dinâmico */
static bool baseline_valid = false;
static float baseline_temp = 0.0f;
static uint32_t baseline_time_ms = 0;
static volatile bool baseline_request = true;
static uint32_t baseline_request_time_ms = 0;

/* TFLM runtime */
static const tflite::Model *model = NULL;
static tflite::MicroInterpreter *interpreter = NULL;
static TfLiteTensor *input = NULL;
static TfLiteTensor *output = NULL;

static constexpr int kTensorArenaSize = 60 * 1024;
static uint8_t tensor_arena[kTensorArenaSize];

/* --------------- Prototypes --------------- */
void core1(void); /* Loop do core1: atualiza OLED e status */
void init_botoes(void); /* Configura BOT_A/B com pull-up */
void init_i2c0(void); /* I2C do AHT20 (sensores) */
void init_i2c1(void); /* I2C do OLED */
void init_oled(void); /* Inicializa display SSD1306 */
void init_aht20(void); /* Reset e init do AHT20 */

void pwm_setup(void); /* Configura PWM do buzzer */
void pwm_on(uint8_t duty_cycle); /* Liga buzzer com duty */
void pwm_off(void); /* Desliga buzzer */

void gpio_irq_handler(uint gpio, uint32_t events); /* ISR dos botões */
int64_t variacao_temp(alarm_id_t, void *user_data); /* Callback do alarme do buzzer */
void wd_errors(uint16_t x, bool positive); /* Contadores de watchdog */

static inline float clamp01(float x); /* Normaliza (atual: passthrough) */
static inline float minmax_scale(float x, int i); /* MinMax por posição */
static void push_temp(float t); /* Empilha temperatura no ring buffer */
static void get_window_ordered(float out[ML_WINDOW]); /* Janela ordenada */

static void ml_init(void); /* Inicializa TFLM */
static bool ml_is_anomaly(const float window[ML_WINDOW], float *out_mse); /* Inferência + MSE */

int main() {
    wd_errors(3, 1);
    stdio_init_all();
    sleep_ms(3000);

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

                bool cond_elevada = false;
                bool cond_severe = false;
                if (win_count >= ML_WINDOW) {
                    float w[ML_WINDOW];
                    get_window_ordered(w);

                    float mse = 0.0f;
                    bool anom = ml_is_anomaly(w, &mse);

                    ml_last_mse = mse;
                    ml_anomaly = anom;

                    mutex_enter_blocking(&data_mutex);
                    snprintf(str_mse, sizeof(str_mse), "MSE:%0.3f", mse);
                    float min10 = w[ML_WINDOW - COND_STABLE_SECONDS];
                    float max10 = min10;
                    float sum10 = 0.0f;
                    for (int i = ML_WINDOW - COND_STABLE_SECONDS; i < ML_WINDOW; i++) {
                        float v = w[i];
                        if (v < min10) {
                            min10 = v;
                        }
                        if (v > max10) {
                            max10 = v;
                        }
                        sum10 += v;
                    }
                    float mean10 = sum10 / (float)COND_STABLE_SECONDS;
                    bool stable10 = (max10 - min10) <= COND_STABLE_DELTA;

                    if (baseline_request &&
                        stable10 &&
                        (now_ms - baseline_request_time_ms) >= (COND_STABLE_SECONDS * 1000)) {
                        baseline_temp = mean10;
                        baseline_time_ms = now_ms;
                        baseline_valid = true;
                        baseline_request = false;
                    } else if (!baseline_valid) {
                        baseline_valid = false;
                    }

                    float delta = w[ML_WINDOW - 1] - baseline_temp;
                    bool cond_elevada = baseline_valid && (fabsf(delta) >= COND_ELEVADA_DELTA);
                    bool cond_severe = baseline_valid &&
                                       (fabsf(delta) >= COND_SEVERA_DELTA) &&
                                       ((now_ms - baseline_time_ms) <= COND_SEVERE_WINDOW_MS);

                    if (baseline_request || !baseline_valid) {
                        snprintf(str_sta, sizeof(str_sta), "CALIB");
                    } else if (cond_severe) {
                        snprintf(str_sta, sizeof(str_sta), "ANOMALIA - SEVERA");
                    } else if (cond_elevada) {
                        snprintf(str_sta, sizeof(str_sta), "ANOMALIA - ELEVADA");
                    } else if (anom) {
                        snprintf(str_sta, sizeof(str_sta), "ANOMALIA - NORMAL");
                    } else {
                        snprintf(str_sta, sizeof(str_sta), "NORMAL");
                    }
                    mutex_exit(&data_mutex);

                    if ((anom || cond_elevada || cond_severe) && !pw.alarm_state && pw.alarm_react) {
                        pw.alarm_pwm = add_alarm_in_ms(3000, variacao_temp, NULL, false);
                        pw.alarm_state = true;
                        pw.alarm_react = false;
                    }

                    if (!(anom || cond_elevada || cond_severe)) {
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
                       (ml_anomaly || cond_elevada || cond_severe) ? "ANOM" : "OK");
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

    /* BOT_A: silencia o alarme atual (rearmazena quando voltar ao normal) */
    if (gpio == BOT_A && (current_time - last_time_a > 300)) {
        pwm_off();
        pw.alarm_state = false;
        pw.alarm_react = false;
        cancel_alarm(pw.alarm_pwm);
        last_time_a = current_time;
    }

    /* BOT_B: solicita recalibração do baseline (CALIB até estabilizar) */
    if (gpio == BOT_B && (current_time - last_time_b > 300)) {
        baseline_request = true;
        baseline_valid = false;
        baseline_request_time_ms = (uint32_t)current_time;
        mutex_enter_blocking(&data_mutex);
        snprintf(str_sta, sizeof(str_sta), "CALIB");
        mutex_exit(&data_mutex);
        last_time_b = current_time;
    }
}

int64_t variacao_temp(alarm_id_t, void *user_data) {
    (void)user_data;
    pwm_on(50);
    return 0;
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

    static float input_copy[ML_WINDOW];
    for (int i = 0; i < ML_WINDOW; i++) {
        float v = minmax_scale(window[i], i);
        input->data.f[i] = v;
        input_copy[i] = v;
    }

    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        printf("❌ Invoke falhou: %d\n", (int)invoke_status);
        return false;
    }

    static bool printed_debug = false;
    if (!printed_debug) {
        printed_debug = true;
        printf("ML dims: in=%d out=%d\n",
               (int)input->dims->data[1], (int)output->dims->data[1]);
        printf("ML bytes: in=%d out=%d\n", (int)input->bytes, (int)output->bytes);
        printf("ML ptrs: in=%p out=%p\n", (void *)input->data.f, (void *)output->data.f);
    }

    float mse = 0.0f;
    for (int i = 0; i < ML_WINDOW; i++) {
        float diff = output->data.f[i] - input_copy[i];
        mse += diff * diff;
    }
    mse /= (float)ML_WINDOW;

    if (out_mse != NULL) {
        *out_mse = mse;
    }
    return (mse > kAnomalyThreshold);
}
