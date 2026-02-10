# Sistema Embarcado de Monitoramento de Temperatura com Detecção de Anomalias (TinyML)

## 📌 Visão Geral

Este projeto implementa um **sistema embarcado de monitoramento ambiental** baseado no **Raspberry Pi Pico / RP2040**, capaz de medir **temperatura e umidade**, exibir informações em um **display OLED**, acionar **alarmes sonoros** e realizar **detecção inteligente de anomalias térmicas** utilizando **TensorFlow Lite for Microcontrollers (TinyML)**.

O sistema foi desenvolvido com foco em **robustez, tempo real e confiabilidade**, utilizando **multicore**, **watchdog**, **mutex**, **interrupções** e **inferência embarcada**.

---

##  Funcionalidades Principais

*  Leitura de **temperatura e umidade** via sensor **AHT20 (I2C)**
*  Interface gráfica em **OLED SSD1306 (I2C)**
*  **Alarme sonoro (buzzer PWM)** para condições anômalas
*  **Detecção de anomalias térmicas** usando **Autoencoder (TinyML)**
*  Execução em **dois núcleos (multicore)** do RP2040
*  Sincronização de dados com **mutex**
*  **Watchdog** com contadores de falha persistentes
*  Botões físicos para **silenciar alarme** e **recalibração do baseline**

---

##  Arquitetura do Sistema

### Multicore

* **Core 0**:

  * Aquisição de dados do sensor
  * Processamento TinyML
  * Lógica de detecção de anomalias
  * Controle de alarmes

* **Core 1**:

  * Atualização do display OLED
  * Exibição de status, MSE e leituras ambientais

### Sincronização

* Uso de **mutex (pico/mutex.h)** para acesso seguro às variáveis compartilhadas entre os núcleos.

---

## Machine Learning Embarcado (TinyML)

* Modelo: **Autoencoder treinado para séries temporais de temperatura**
* Framework: **TensorFlow Lite for Microcontrollers**
* Janela temporal: **60 amostras**
* Técnica de normalização: **Min-Max por posição da janela**
* Métrica de decisão: **Erro Quadrático Médio (MSE)**
* Threshold de anomalia: `kAnomalyThreshold`

### Classificação de Estados

* **NORMAL**: comportamento esperado
* **ANOMALIA - NORMAL**: detectada via ML
* **ANOMALIA - ELEVADA**: desvio significativo do baseline
* **ANOMALIA - SEVERA**: desvio crítico em curto intervalo
* **CALIB**: recalibração automática do baseline

---

## Sistema de Alarme

* Buzzer controlado por **PWM**
* Alarme ativado automaticamente em condições anômalas
* Botão dedicado para **silenciar o alarme**
* Reativação automática quando o sistema retorna ao estado normal

---

##  Confiabilidade e Segurança

* **Watchdog habilitado (8s)** para evitar travamentos
* Contadores de erro persistentes via **watchdog scratch registers**
* Monitoramento de vida dos dois núcleos (heartbeat)
* Reinicialização segura em caso de falhas críticas

---

##  Interface Física

### Botões

* **BOT_A**: Silencia o alarme atual
* **BOT_B**: Solicita recalibração do baseline térmico

### Display OLED

Exibe:

* Temperatura (°C)
* Umidade (%)
* Status do sistema
* Valor de MSE do modelo ML

---

##  Tecnologias Utilizadas

* Linguagem: **C/C++**
* MCU: **RP2040 (Raspberry Pi Pico)**
* ML: **TensorFlow Lite for Microcontrollers**
* Sensores: **AHT20**
* Display: **SSD1306 OLED**
* Comunicação: **I2C**

---

##  Aplicações Potenciais

* Monitoramento térmico industrial
* Sistemas de segurança e prevenção de falhas
* Ambientes críticos (laboratórios, data centers)
* IoT embarcado com inteligência local
* Projetos acadêmicos e pesquisa em TinyML
