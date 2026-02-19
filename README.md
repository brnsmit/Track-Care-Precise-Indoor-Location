# Track&Care – Localização Indoor de Pacientes com Machine Learning

---

## 1. Visão Geral

O projeto Track&Care tem como objetivo classificar a localização indoor de pacientes em uma clínica de repouso utilizando sinais BLE (Bluetooth Low Energy) captados por smartphones da equipe.

O desafio principal não é apenas treinar um modelo de classificação, mas lidar com um cenário realista:

* Sinais ruidosos
* Dispositivos heterogêneos
* Dados inconsistentes
* Necessidade de generalização para novos aparelhos

A solução proposta foi estruturada com foco em robustez, validação realista e viabilidade de deploy.

---

## 2. O Problema

O RSSI (Received Signal Strength Indicator) é um sinal extremamente instável. Ele varia por:

* Distância
* Obstáculos físicos
* Interferência e multipath
* Orientação do dispositivo
* Modelo do smartphone (calibração de antena e chipset)

Além disso:

* A equipe é dinâmica (novos IMEIs entram no sistema).
* O modelo precisa funcionar com dispositivos não vistos durante o treino.
* O dataset contém ruído inserido propositalmente.

O desafio real é transformar esse sinal “sujo” em uma classificação de sala confiável.

---

## 3. Dataset

Arquivo utilizado:
`telemetria_track_care_treino.csv`

### Principais colunas

* `timestamp`
* `patient_id`
* `imei`
* `employee_id`
* `sensor_latlong`
* `sensor_room`
* `rssi`
* `actual_patient_room` (target)

---

## 4. Análise dos Dados e Tratamento de Ruído

Durante a análise exploratória, alguns pontos críticos foram identificados:

### 4.1 Valores de RSSI fisicamente impossíveis

Foram encontrados valores de RSSI fora do intervalo típico para BLE indoor (ex: acima de -20 dBm ou abaixo de -100 dBm).

Esses valores foram removidos por serem fisicamente improváveis e provavelmente decorrentes de erro de simulação ou ruído extremo.

```python
df = df[(df['rssi'] >= -100) & (df['rssi'] <= -20)]
```

---

### 4.2 IMEI artificial com baixa representatividade

Foi identificado um dispositivo:

```
IMEI_999999
```

Com apenas ~75 amostras, enquanto os demais possuíam ~1500.

Esse padrão indica:

* Dispositivo sintético
* Possível ruído proposital
* Alto risco de distorcer estatísticas

Decisão tomada:

Remover IMEIs com menos de 100 amostras para evitar distorção de distribuição.

```python
imei_counts = df['imei'].value_counts()
valid_imeis = imei_counts[imei_counts > 100].index
df = df[df['imei'].isin(valid_imeis)]
```

---

### 4.3 Viés estrutural de hardware

Dispositivos diferentes apresentavam distribuições de RSSI significativamente distintas.

Sem tratamento, o modelo poderia:

* Memorizar padrões específicos de IMEI
* Superestimar performance
* Falhar com novos dispositivos

Para mitigar isso, foram adotadas duas estratégias:

1. Normalização estatística por IMEI (quando aplicável)
2. Validação Leave-One-IMEI-Out

---

## 5. Engenharia de Features

O modelo não utiliza apenas RSSI bruto.

Foi aplicado janelamento temporal com agregação estatística.

### Janelamento Temporal

Janela de 2 minutos para agregação de leituras por paciente.

Para cada janela foram calculadas:

* Média do RSSI
* Desvio padrão
* Mínimo
* Máximo
* Contagem de leituras
* Média de latitude
* Média de longitude

Exemplo:

```python
agg_df = (
    df.groupby(['patient_id', 'window'])
    .agg({
        'rssi': ['mean','std','min','max','count'],
        'sensor_lat': 'mean',
        'sensor_long': 'mean',
        'actual_patient_room': lambda x: x.mode()[0],
        'imei': 'first'
    })
)
```

Motivação:

Uma única leitura de RSSI é extremamente instável.
A agregação temporal reduz variância e melhora a robustez.

---

## 6. Estratégia de Validação

Foi utilizada validação Leave-One-IMEI-Out.

Para cada iteração:

* Treino com N-1 dispositivos
* Teste em 1 dispositivo nunca visto

Isso simula o cenário real de novos smartphones entrando no sistema.

```python
for test_imei in unique_imeis:
    train_mask = groups != test_imei
    test_mask = groups == test_imei
```

Essa abordagem evita overfitting por hardware.

---

## 7. Modelo Utilizado

Algoritmo: XGBoost Classifier

Motivos da escolha:

* Bom desempenho com dados tabulares
* Controle explícito de regularização
* Robustez a ruído
* Inferência rápida
* Feature importance interpretável

Principais hiperparâmetros:

```yaml
n_estimators: 200
max_depth: 4
learning_rate: 0.05
subsample: 0.8
colsample_bytree: 0.8
```

Esses valores foram escolhidos para evitar sobreajuste, dado o número reduzido de IMEIs.

---

## 8. Resultados

Validação Leave-One-IMEI-Out mostrou que:

* O modelo supera significativamente o baseline aleatório.
* A agregação temporal melhora estabilidade.
* Ainda existe limitação estrutural pela baixa diversidade de hardware (apenas 3 IMEIs válidos).

Esse ponto é importante:

O dataset é pequeno para generalização forte.
Em ambiente real, seria necessário coletar dados de mais dispositivos.

---

## 9. API de Inferência

Endpoint: `POST /predict`

Entrada:

```json
{
  "imei": "IMEI_IPHONE_X",
  "sensor_latlong": "-22.809788, -47.059685",
  "sensor_room": "Enfermaria",
  "rssi": -71.76,
  "patient_id": "PAT_001",
  "timestamp": "2026-02-10T08:00:00"
}
```

Saída:

```json
{
  "predicted_room": "Quarto 1",
  "confidence": 0.87
}
```

---

## 10. Justificativas Técnicas

### 10.1 Escolha do Janelamento

* 2 minutos → baixa latência, maior responsividade
* 5 minutos → maior estabilidade, porém menor sensibilidade a mudança rápida

Para ambiente clínico, 5 minutos é mais robusto.
Para resposta rápida (quase tempo real), 2 minutos é aceitável.

---

### 10.2 Uso de IMEI

IMEI não é utilizado como feature categórica.

Ele é usado apenas para:

* Agrupamento na validação
* Normalização estatística

Isso evita que o modelo memorize dispositivos específicos.

---

### 10.3 employee_id

Optou-se por não utilizar `employee_id` como feature principal, pois pode introduzir vazamento indireto (modelo aprender comportamento do funcionário em vez do sinal físico).

---

## 11. Próximos Passos

* Implementar agregação temporal diretamente na API
* Monitorar drift de distribuição de RSSI por IMEI
* Coletar dados de maior diversidade de dispositivos
* Avaliar ensemble leve (XGBoost + modelo linear)

## Autor

- **Bruno Paiva Smit de Freitas** - [brnsmit](https://github.com/brnsmit)


