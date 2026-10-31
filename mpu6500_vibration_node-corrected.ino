/*
  ESP32 + MPU6500 (GY-521 clone) Vibration Feature Extraction
  Continuous streaming — no buttons, no modes, no session UI.
  Outputs human-readable feature lines over USB serial at 115200 baud.

  Hardware:
    MPU6500 on I2C (GPIO21=SDA, GPIO22=SCL)
    SSD1306 OLED 128x64 on I2C 0x3C

  Libraries: ArduinoFFT (v2), Adafruit SSD1306, Adafruit GFX
*/

#include <Wire.h>
#include <arduinoFFT.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

// =====================================================================
//  MPU / OLED CONFIG
// =====================================================================
#define MPU_ADDR        0x68
#define WHO_AM_I_REG    0x75
#define PWR_MGMT_1      0x6B
#define ACCEL_CONFIG2   0x1D
#define ACCEL_XOUT_H    0x3B

#define OLED_ADDR       0x3C
#define OLED_WIDTH      128
#define OLED_HEIGHT     64
#define OLED_RESET      -1

#define SAMPLES         256
#define ACCEL_SENS      16384.0f
#define I2C_CLOCK_HZ    100000

#define FIFO_EN_REG     0x23
#define USER_CTRL_REG   0x6A
#define FIFO_COUNT_H    0x72
#define FIFO_R_W        0x74
#define BURST_BYTES     120

// =====================================================================
//  HARDWARE OBJECTS
// =====================================================================
Adafruit_SSD1306 display(OLED_WIDTH, OLED_HEIGHT, &Wire, OLED_RESET);

// =====================================================================
//  FFT BUFFERS — separate per-axis, never combined
// =====================================================================
double xData[SAMPLES], xImag[SAMPLES];
double yData[SAMPLES], yImag[SAMPLES];
double zData[SAMPLES], zImag[SAMPLES];
ArduinoFFT<double> FFT_X(xData, xImag, SAMPLES, 1000.0);
ArduinoFFT<double> FFT_Y(yData, yImag, SAMPLES, 1000.0);
ArduinoFFT<double> FFT_Z(zData, zImag, SAMPLES, 1000.0);

// =====================================================================
//  STRUCTS
// =====================================================================
struct FeatureSet {
  float rms, peak, crestFactor, kurtosis, skewness;
  float dominantFreq, spectralEntropy;
};
struct AxisOffsets { float x, y, z; };

// =====================================================================
//  GLOBALS
// =====================================================================
uint32_t    cycleCount    = 0;
uint32_t    i2cErrorCount = 0;
AxisOffsets gAxisOffset   = {0, 0, 0};

// =====================================================================
//  I2C HELPERS — UNCHANGED
// =====================================================================
bool writeRegister(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(val);
  uint8_t err = Wire.endTransmission();
  if (err != 0) {
    i2cErrorCount++;
    Serial.printf("I2C write FAIL reg 0x%02X code %d\n", reg, err);
    return false;
  }
  return true;
}

bool readRegister(uint8_t reg, uint8_t &val) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  uint8_t err = Wire.endTransmission(false);
  if (err != 0) { i2cErrorCount++; return false; }
  uint8_t got = Wire.requestFrom(MPU_ADDR, 1, true);
  if (got != 1) { i2cErrorCount++; return false; }
  val = Wire.read();
  return true;
}

bool probeMPU() {
  uint8_t who;
  if (!readRegister(WHO_AM_I_REG, who)) return false;
  return (who == 0x68 || who == 0x70);
}

bool mpuInit() {
  bool ok = true;
  ok &= writeRegister(PWR_MGMT_1, 0x00);
  delay(50);
  ok &= writeRegister(ACCEL_CONFIG2, 0x08);
  delay(10);
  return ok;
}

bool enableFIFO() {
  bool ok = true;
  ok &= writeRegister(USER_CTRL_REG, 0x04);
  delay(5);
  ok &= writeRegister(FIFO_EN_REG, 0x08);
  ok &= writeRegister(USER_CTRL_REG, 0x40);
  return ok;
}

bool readFIFOCount(uint16_t &count) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(FIFO_COUNT_H);
  uint8_t err = Wire.endTransmission(false);
  if (err != 0) { i2cErrorCount++; return false; }
  uint8_t got = Wire.requestFrom(MPU_ADDR, 2, true);
  if (got != 2) { i2cErrorCount++; return false; }
  uint16_t h = Wire.read();
  uint16_t l = Wire.read();
  count = (h << 8) | l;
  return true;
}

bool readFIFOBurst(uint8_t *buf, uint16_t nBytes) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(FIFO_R_W);
  uint8_t err = Wire.endTransmission(false);
  if (err != 0) { i2cErrorCount++; return false; }
  uint8_t got = Wire.requestFrom(MPU_ADDR, (int)nBytes, true);
  if (got != nBytes) { i2cErrorCount++; return false; }
  for (uint16_t i = 0; i < nBytes; i++) buf[i] = Wire.read();
  return true;
}

bool readAccelRaw(int16_t &ax, int16_t &ay, int16_t &az) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(ACCEL_XOUT_H);
  uint8_t err = Wire.endTransmission(false);
  if (err != 0) { i2cErrorCount++; return false; }
  uint8_t got = Wire.requestFrom(MPU_ADDR, 6, true);
  if (got != 6) { i2cErrorCount++; return false; }
  ax = (Wire.read() << 8) | Wire.read();
  ay = (Wire.read() << 8) | Wire.read();
  az = (Wire.read() << 8) | Wire.read();
  return true;
}

inline float toG(int16_t raw) { return raw / ACCEL_SENS; }

// =====================================================================
//  ACQUISITION — UNCHANGED
// =====================================================================
float acquireBlock() {
  uint8_t buf[BURST_BYTES];
  int samplesCollected = 0;
  unsigned long t0 = micros();

  while (samplesCollected < SAMPLES) {
    uint16_t avail;
    if (!readFIFOCount(avail)) continue;
    if (avail < 6) continue;

    int remainingBytes = (SAMPLES - samplesCollected) * 6;
    uint16_t wantBytes = min((int)(avail - (avail % 6)), min((int)BURST_BYTES, remainingBytes));
    if (wantBytes < 6) continue;

    if (!readFIFOBurst(buf, wantBytes)) continue;

    for (uint16_t i = 0; i < wantBytes; i += 6) {
      int16_t ax = (buf[i]     << 8) | buf[i + 1];
      int16_t ay = (buf[i + 2] << 8) | buf[i + 3];
      int16_t az = (buf[i + 4] << 8) | buf[i + 5];

      xData[samplesCollected] = toG(ax) - gAxisOffset.x;
      yData[samplesCollected] = toG(ay) - gAxisOffset.y;
      zData[samplesCollected] = toG(az) - gAxisOffset.z;
      xImag[samplesCollected] = 0.0;
      yImag[samplesCollected] = 0.0;
      zImag[samplesCollected] = 0.0;

      samplesCollected++;
    }
  }

  unsigned long elapsedUs = micros() - t0;
  return SAMPLES / (elapsedUs / 1000000.0f);
}

// =====================================================================
//  CALIBRATION — UNCHANGED
// =====================================================================
AxisOffsets calibrateAxisOffsets() {
  AxisOffsets off = {0, 0, 0};
  int16_t ax, ay, az;
  double sx = 0, sy = 0, sz = 0;
  const int N = 200;
  int collected = 0;
  while (collected < N) {
    if (!readAccelRaw(ax, ay, az)) { delay(2); continue; }
    sx += toG(ax); sy += toG(ay); sz += toG(az);
    collected++;
    delay(2);
  }
  off.x = sx / N; off.y = sy / N; off.z = sz / N;
  return off;
}

// =====================================================================
//  FEATURE EXTRACTION — UNCHANGED
// =====================================================================
FeatureSet computeAxisFeatures(double *data, double *imagData,
                                ArduinoFFT<double> &fft, float actualFs) {
  FeatureSet f;

  float mean = 0;
  for (int i = 0; i < SAMPLES; i++) mean += data[i];
  mean /= SAMPLES;

  float m2 = 0, m3 = 0, m4 = 0, sumSq = 0, peak = 0;
  for (int i = 0; i < SAMPLES; i++) {
    float d = data[i] - mean;
    m2 += d * d; m3 += d * d * d; m4 += d * d * d * d;
    sumSq += data[i] * data[i];
    if (fabs(data[i]) > peak) peak = fabs(data[i]);
  }
  m2 /= SAMPLES; m3 /= SAMPLES; m4 /= SAMPLES;

  float stdDev = sqrt(m2);
  f.rms         = sqrt(sumSq / SAMPLES);
  f.peak        = peak;
  f.crestFactor = (f.rms > 1e-6) ? f.peak / f.rms : 0;
  f.skewness    = (stdDev > 1e-6) ? m3 / pow(stdDev, 3) : 0;
  f.kurtosis    = (stdDev > 1e-6) ? (m4 / pow(stdDev, 4)) - 3.0 : 0;

  fft.windowing(FFTWindow::Hamming, FFTDirection::Forward);
  fft.compute(FFTDirection::Forward);
  fft.complexToMagnitude();

  int nBins = SAMPLES / 2;
  float freqRes = actualFs / SAMPLES;

  float maxMag = 0, totalPower = 0;
  int maxBin = 1;
  for (int i = 1; i < nBins; i++) {
    float power = data[i] * data[i];
    totalPower += power;
    if (data[i] > maxMag) { maxMag = data[i]; maxBin = i; }
  }
  f.dominantFreq = maxBin * freqRes;

  float entropy = 0;
  if (totalPower > 1e-9) {
    for (int i = 1; i < nBins; i++) {
      float p = (data[i] * data[i]) / totalPower;
      if (p > 1e-12) entropy -= p * log(p);
    }
    entropy /= log((float)(nBins - 1));
  }
  f.spectralEntropy = entropy;

  return f;
}

// =====================================================================
//  SERIAL REPORTING — UNCHANGED
// =====================================================================
void printAxisBlock(const char *label, const FeatureSet &f) {
  Serial.printf("[%s] RMS=%.4fg Peak=%.4fg Crest=%.3f Kurt=%.3f"
                " Skew=%.3f DomFreq=%.1fHz Entropy=%.3f\n",
                label, f.rms, f.peak, f.crestFactor,
                f.kurtosis, f.skewness, f.dominantFreq, f.spectralEntropy);
}

void reportSerial(const FeatureSet &fx, const FeatureSet &fy,
                  const FeatureSet &fz, float actualFs) {
  Serial.println(F("---------------------------------------"));
  Serial.printf("Cycle #%lu | Fs=%.1fHz | I2C errors(total)=%lu\n",
                cycleCount, actualFs, i2cErrorCount);
  printAxisBlock("X", fx);
  printAxisBlock("Y", fy);
  printAxisBlock("Z", fz);
}

// =====================================================================
//  OLED DISPLAY — cycles through axes/features every 2 windows
// =====================================================================
void reportOLED(const FeatureSet &fx, const FeatureSet &fy,
                const FeatureSet &fz, float actualFs) {
  const FeatureSet *axes[3] = { &fx, &fy, &fz };
  const char *axisNames[3]  = { "X", "Y", "Z" };

  int page    = cycleCount % 6;
  int axisIdx = page / 2;
  bool subA   = (page % 2 == 0);
  const FeatureSet &f = *axes[axisIdx];

  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);

  display.setCursor(0, 0);
  display.printf("Axis:%s Fs:%.0fHz", axisNames[axisIdx], actualFs);
  display.setCursor(0, 9);
  display.printf("Err:%lu  #%lu", i2cErrorCount, cycleCount);
  display.drawLine(0, 18, 128, 18, SSD1306_WHITE);
  display.setCursor(0, 22);

  if (subA) {
    display.printf("RMS : %.3f g\n", f.rms);
    display.printf("Peak: %.3f g\n", f.peak);
    display.printf("Crst: %.3f\n",   f.crestFactor);
    display.printf("Kurt: %.3f\n",   f.kurtosis);
  } else {
    display.printf("Skew: %.3f\n",    f.skewness);
    display.printf("Freq: %.1f Hz\n", f.dominantFreq);
    display.printf("Entr: %.3f\n",    f.spectralEntropy);
  }
  display.display();
}

// =====================================================================
//  SETUP
// =====================================================================
void setup() {
  Serial.begin(115200);
  delay(300);

  Wire.begin();
  Wire.setClock(I2C_CLOCK_HZ);
  Serial.printf("I2C_CLOCK_HZ: %d\n", I2C_CLOCK_HZ);

  if (!probeMPU()) {
    Serial.println(F("FATAL: MPU not found at 0x68 (WHO_AM_I 0x68/0x70). Halting."));
    while (true) delay(1000);
  }
  if (!mpuInit()) {
    Serial.println(F("FATAL: MPU init failed (I2C write error). Halting."));
    while (true) delay(1000);
  }
  if (!enableFIFO()) {
    Serial.println(F("FATAL: FIFO enable failed (I2C write error). Halting."));
    while (true) delay(1000);
  }
  Serial.println(F("MPU6500(clone) initialized, DLPF bypassed, FIFO enabled."));

  if (!display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR)) {
    Serial.println(F("WARNING: OLED not found at 0x3C. Serial-only mode."));
  } else {
    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);
    display.setCursor(0, 0);
    display.println(F("Calibrating..."));
    display.println(F("Keep sensor still."));
    display.display();
  }

  Serial.println(F("Calibrating per-axis offsets (keep sensor still ~1s)..."));
  gAxisOffset = calibrateAxisOffsets();
  Serial.printf("Axis offsets: X=%.4fg Y=%.4fg Z=%.4fg\n",
                gAxisOffset.x, gAxisOffset.y, gAxisOffset.z);

  display.clearDisplay();
  display.setCursor(0, 0);
  display.println(F("Ready."));
  display.println(F("Streaming data..."));
  display.display();
  delay(500);
}

// =====================================================================
//  LOOP — continuous acquire → compute → report
// =====================================================================
void loop() {
  float actualFs = acquireBlock();
  FeatureSet fx  = computeAxisFeatures(xData, xImag, FFT_X, actualFs);
  FeatureSet fy  = computeAxisFeatures(yData, yImag, FFT_Y, actualFs);
  FeatureSet fz  = computeAxisFeatures(zData, zImag, FFT_Z, actualFs);
  reportSerial(fx, fy, fz, actualFs);
  reportOLED(fx, fy, fz, actualFs);
  cycleCount++;
}
