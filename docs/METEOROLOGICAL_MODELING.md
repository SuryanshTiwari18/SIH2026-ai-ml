# Meteorological & Physical Modeling Formulation

This document outlines the governing equations, thermodynamic constraints, and climatological foundations behind the SkyGuard AI synthetic AWS sensor telemetry generator.

---

## 1. Elevation & Barometric Pressure (ISA Standard)

Surface atmospheric pressure $P(h)$ drops non-linearly with altitude $h$ above mean sea level. Rather than using arbitrary values, station surface pressure baselines are computed using the **International Standard Atmosphere (ISA)** barometric formula:

$$P_{\text{base}}(h) = P_0 \left(1 - \frac{L \cdot h}{T_0}\right)^{\frac{g \cdot M}{R_0 \cdot L}}$$

Where:
- $P_0 = 1013.25\text{ hPa}$ (standard sea-level pressure)
- $T_0 = 288.15\text{ K}$ (standard sea-level temperature = 15°C)
- $L = 0.0065\text{ K/m}$ (standard tropospheric lapse rate)
- $g = 9.80665\text{ m/s}^2$ (gravitational acceleration)
- $M = 0.0289644\text{ kg/mol}$ (molar mass of dry air)
- $R_0 = 8.3144598\text{ J/(mol}\cdot\text{K)}$ (universal gas constant)
- Exponent: $\frac{g \cdot M}{R_0 \cdot L} \approx 5.25588$

### Station Elevations & Baselines:
- **Mumbai (Colaba)**: $h = 11\text{ m} \implies P_{\text{base}} \approx 1011.9\text{ hPa}$
- **New Delhi (Safdarjung)**: $h = 211\text{ m} \implies P_{\text{base}} \approx 988.3\text{ hPa}$
- **Nagpur**: $h = 310\text{ m} \implies P_{\text{base}} \approx 977.0\text{ hPa}$
- **Shillong**: $h = 1496\text{ m} \implies P_{\text{base}} \approx 845.5\text{ hPa}$
- **Shimla**: $h = 2205\text{ m} \implies P_{\text{base}} \approx 776.4\text{ hPa}$

---

## 2. Semi-Diurnal Atmospheric Thermal Tides ($S_2$)

In tropical and subtropical latitudes (India spans $\approx 8^\circ\text{N} - 36^\circ\text{N}$), solar absorption by stratospheric ozone and tropospheric water vapor drives a pronounced 12-hour barometric oscillation. The semi-diurnal thermal tide $S_2(P)$ is modeled as:

$$P_{\text{tide}}(t) = A \cos(\phi) \cdot \cos\left(\frac{2\pi (t_{\text{solar}} - 10.0)}{12.0}\right)$$

Where:
- $A \approx 1.65\text{ hPa}$
- $\phi$ is the station latitude
- Peaks occur consistently around **10:00** and **22:00** local solar time
- Troughs occur around **04:00** and **16:00** local solar time

---

## 3. Asymmetric Solar Diurnal Cycles

Surface air temperature exhibits an asymmetric cycle driven by incoming solar insolation and lagged ground heating:
- **Trough (Minimum)**: Just before sunrise ($\approx \text{05:30 IST}$).
- **Peak (Maximum)**: Mid-afternoon ($\approx \text{14:30 IST}$), reflecting a 2.5-hour thermal inertia lag after solar noon.

The diurnal temperature curve $D_T(t)$ is modeled using shifted Fourier modes:

$$f(t) = \cos\left(\frac{2\pi (t_{\text{solar}} - 14.5)}{24}\right) + 0.22 \cos\left(\frac{4\pi (t_{\text{solar}} - 14.5)}{24} - 0.4\right)$$

Normalized to $[-0.5, +0.5]$ and scaled by the station's climatological Diurnal Temperature Range ($\text{DTR}$):

$$T_{\text{diurnal}}(t) = \text{DTR} \cdot D_T(t)$$

### Diurnal Relative Humidity Dynamics:
Relative humidity has an inverse thermodynamic response to the diurnal temperature cycle. As temperature rises during the day, saturation vapor pressure $e_s(T)$ increases exponentially, depressing RH:

$$RH_{\text{diurnal}}(t) = -\text{DRR} \cdot D_T(t)$$

Where $\text{DRR}$ is the station's diurnal humidity range.

---

## 4. August-Roche-Magnus Approximation & Thermodynamic Consistency

The saturation vapor pressure $e_s(T)$ in hPa is calculated using the August-Roche-Magnus equation:

$$e_s(T) = 6.112 \cdot \exp\left(\frac{17.67 \cdot T}{T + 243.5}\right)$$

Actual vapor pressure:
$$e(T, RH) = e_s(T) \cdot \frac{RH}{100.0}$$

The dew point temperature $T_d$ is derived as:
$$\gamma(T, RH) = \ln\left(\frac{e}{6.112}\right) = \frac{17.67 \cdot T}{T + 243.5} + \ln\left(\frac{RH}{100.0}\right)$$
$$T_d = \frac{243.5 \cdot \gamma}{17.67 - \gamma}$$

### Physical Invariants:
1. $0\% \le RH \le 100\%$
2. $RH \le 100\% \implies \ln(RH/100) \le 0 \implies \gamma \le \frac{17.67 T}{T + 243.5} \implies T_d \le T_{\text{air}}$

Any reading where $T_d > T_{\text{air}}$ or $RH > 100\%$ is an immediate physical impossibility.

---

## 5. Synoptic Coupling (Ornstein-Uhlenbeck / AR(1) Process)

Multi-day weather systems (monsoon lows, troughs, high-pressure ridges) are simulated via an Ornstein-Uhlenbeck continuous-time autoregressive process:

$$X_{t} = \phi X_{t-1} + \sigma \sqrt{1 - \phi^2} \cdot \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, 1)$$

Where $\phi = \exp(-\Delta t / \tau)$ with $\tau \approx 3.5\text{ days}$.

The synoptic pressure anomaly is physically coupled to temperature and moisture:
- **Low Pressure System ($X_t < 0$)**: Influx of cyclonic moisture ($\Delta RH = -2.2 X_t > 0$), cloud cover suppresses daytime solar heating, reducing effective diurnal temperature range.
- **High Pressure Ridge ($X_t > 0$)**: Clear skies, strong daytime solar heating, lower humidity.
