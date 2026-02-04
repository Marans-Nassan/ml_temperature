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
#include "lib/ssd1306.h"
#include "lib/font.h"
#include "lib/aht20.h"
#include "hardware/watchdog.h"
#include "hardware/structs/watchdog.h"

#define I2C_PORT_A i2c0
#define I2C_SDA_A 0
#define I2C_SCL_A 1
#define BOT_A 5
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
char str_tmp[5];  
char str_umi[5]; 
typedef struct checks{
    bool error1;
    bool error2;
    bool error3;
} checks; 
checks c = {false, false, false};

typedef struct WD_checks{
    uint16_t error001;
    uint16_t error002;
    uint16_t error003;
} WD_checks;
WD_checks wdc = {0, 0, 0};

uint8_t slice = 0;
typedef struct {
    float dc;           // wrap value do PWM
    float div;          // divisor de clock
    bool alarm_state;   // estado da sirene
    bool alarm_react;   // permite reativar alarme
    alarm_id_t alarm_pwm;
} pwm_struct;
pwm_struct pw = {7812.5, 32.0, false, true, 0};

mutex_t data_mutex;

void core1(void); // Função executada no segundo núcleo (core1) do RP2040. Responsável por atualizar o display OLED continuamente.
void init_botoes(void); // Inicializa os botões como entrada digital com pull-up interno.
void init_i2c0(void); // Inicializa a interface I2C0 (usada pelos sensores BMP280 e AHT20).
void init_i2c1(void); // Inicializa a interface I2C1 (usada pelo display OLED).
void init_oled(void); // Inicializa o display OLED SSD1306.
void init_aht20(void); // Inicializa o sensor AHT20, incluindo reset e configuração inicial.
void pwm_setup(void); // Configura o PWM do buzzer (frequência e wrap).
void pwm_on(uint8_t duty_cycle); // Ativa o PWM do buzzer com duty cycle especificado.
void pwm_off(void); // Desliga o PWM do buzzer e garante que não haja vazamento de sinal.
void gpio_irq_handler(uint gpio, uint32_t events); // Handler de interrupção para o botão. É usado para desligar o alarme quando bot_a é pressionado.
int64_t variacao_temp(alarm_id_t, void *user_data); // Callback de alarme usado pelo PWM quando a temperatura está fora da faixa. Ativa o buzzer e retorna 1 para manter o alarme ativo.
void wd_errors(uint16_t x, bool positive); // Verifica onde foi que ativou o watchdog.


int main() {
    stdio_init_all();
    if (watchdog_caused_reboot()) {
        uint32_t reset_count = watchdog_hw->scratch[0]; // scratch mantém valor entre resets por watchdog
        reset_count++;
        watchdog_hw->scratch[0] = reset_count;
        printf("\n\n>>> Reiniciado pelo Watchdog! Contagem de resets: %d\n", reset_count);
        printf("Erro001 - %d", wdc.error001);
        printf("Erro002 - %d", wdc.error002);
        printf("Erro003 - %d", wdc.error003);
    } else {
        printf(">>> Reset normal (Power On). Iniciando contador em 0.\n");
        // watchdog_hw->scratch[0] = 0;
    }
    watchdog_enable(4000, true); //Sistema watchdog contra falhas. Caso o programa trave por mais de 4 segundos sem o watchdog_update resetar, o programa é forçado a reiniciar.
    mutex_init(&data_mutex);
    multicore_launch_core1(core1);
    init_botoes();
    init_i2c0();
    init_aht20();
    pwm_setup();
    
    gpio_set_irq_enabled_with_callback(BOT_A, GPIO_IRQ_EDGE_FALL, true, &gpio_irq_handler);   

    while (true) {
        wd_errors(1,1);
        // Leitura do AHT20
        if (aht20_read(I2C_PORT_A, &data)) {
            c.error3 = true;
            printf("Temperatura AHT: %.2f C\n", data.temperature);
            printf("Umidade: %.2f %%\n\n\n", data.humidity);
            sprintf(str_tmp, "%.1fC", data.temperature);
            sprintf(str_umi, "%.1f%%", data.humidity);
        } else {
            printf("Erro na leitura do AHT10!\n\n\n");
            c.error3 = false;
        }
        if(!c.error2 && !c.error3) {c.error1 = true;}
        else {c.error1 = false;}

        mutex_exit(&data_mutex);
 
        if ((temperature < 1000 || temperature > 4000) && !pw.alarm_state && pw.alarm_react) {
            pw.alarm_pwm = add_alarm_in_ms(3000, variacao_temp, NULL, false);
            pw.alarm_state = true;
            pw.alarm_react = false;
        }
        if (temperature >= 1000 && temperature <= 4000) {
            if (pw.alarm_state) {
                pwm_off();
                pw.alarm_state = false;
                cancel_alarm(pw.alarm_pwm);
            }
            pw.alarm_react = true;
        }
        media = ((temperature/100.0) + data.temperature)/2 ;

        sleep_ms(1);
        watchdog_update();
        wd_errors(1,0); 
    }
}


void core1(void) {
    sleep_ms(5000);
    bool cor = true;
    init_i2c1();
    init_oled();
    
    while(true) {
        ssd1306_fill(&ssd, !cor);                          
        ssd1306_rect(&ssd, 3, 3, 122, 60, cor, !cor);      
        ssd1306_line(&ssd, 3, 38, 123, 38, cor); // Linha Umidade e Temp Cima 
        ssd1306_line(&ssd, 3, 50, 123, 50, cor); // Linha Umidade e Temp Baixo
        ssd1306_line(&ssd, 3, 15, 123, 15, cor); // Linha Data e Hora Cima
        ssd1306_line(&ssd, 3, 27, 123, 27, cor); // Linha Data e Hora Baixo              
        ssd1306_draw_string(&ssd, "Data:", 7, 6);
        ssd1306_draw_string(&ssd, "Hora:", 7, 18);  
        ssd1306_draw_string(&ssd, " UMI    TEMP", 10, 41); 
        ssd1306_draw_string(&ssd, "Alarme:", 7, 30);  
        ssd1306_line(&ssd, 63, 39, 63, 60, cor); // Linha Central         
 
        mutex_enter_blocking(&data_mutex);                
        if(c.error3) {
            ssd1306_draw_string(&ssd, str_tmp, 73, 52);             
            ssd1306_draw_string(&ssd, str_umi, 13, 52);
        } else {
            ssd1306_draw_string(&ssd, "---", 73, 52);
            ssd1306_draw_string(&ssd, "---", 13, 52);
        }
        mutex_exit(&data_mutex);
        ssd1306_send_data(&ssd);
        sleep_ms(1000);
    }
}

void init_botoes(void) {
    for (uint8_t botoes = 5; botoes < 7; botoes++){
        gpio_init(botoes);
        gpio_set_dir(botoes, GPIO_IN);
        gpio_pull_up(botoes);
    }
}

void init_i2c0(void) {
    i2c_init(I2C_PORT_A, 400*1000);
    for(uint8_t init_i2c0 = 0 ; init_i2c0 < 2; init_i2c0 ++){
        gpio_set_function(init_i2c0, GPIO_FUNC_I2C);
        gpio_pull_up(init_i2c0);
    }
}

void init_i2c1(void) {
    i2c_init(I2C_PORT_B, 400*1000);
    for(uint8_t init_i2c1 = 14 ; init_i2c1 < 16; init_i2c1 ++){
        gpio_set_function(init_i2c1, GPIO_FUNC_I2C);
        gpio_pull_up(init_i2c1);
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
    uint64_t current_time = to_ms_since_boot(get_absolute_time());
    static uint64_t last_time_a = 0, last_time_b = 0;
    if(gpio == BOT_A && (current_time - last_time_a > 300)) {
        pwm_off();
        pw.alarm_state = false;
        pw.alarm_react = false;
        cancel_alarm(pw.alarm_pwm);
        last_time_a = current_time;
    }
}

int64_t variacao_temp(alarm_id_t, void *user_data) {
    pwm_on(50);
    return 1;
}

void wd_errors (uint16_t x, bool positive){
    switch(x) {
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
    }
}