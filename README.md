
# Dream Learning with Dynamic Textual Prompts Creates Novel Paradigm for Pan-Arctic Sea Ice Seasonal Forecasting
## IceDreamer: A Language-Infused Arctic Sea Ice Forecasting Framework

### 1. Abstract
Arctic sea ice, a key regulator of Earth’s climate system, has undergone unprecedented retreat due to global warming—making accurate seasonal forecasting critical for climate research and policymaking. Traditional numerical models face challenges of computational intensity and parameterization uncertainties, while existing machine learning approaches struggle to integrate multimodal data and domain expertise, leading to under-exploited complex spatiotemporal relationships. To address these limitations, this study introduces **IceDreamer**, a language-infused learning framework that combines a dynamic textual prompt corpus with observational data via an innovative Dream Learning paradigm.  

Inspired by human memory consolidation during sleep, IceDreamer adopts a self-supervised "dream–wake" cycle and a quantum-entanglement-like feature interaction module (Dream Weaver Entangler, DWE) to expand the learning sample space and capture nonlinear spatiotemporal dependencies. By leveraging large language models (LLMs) to inject physical knowledge from textual prompts, IceDreamer bridges the gap between physics-based and data-driven forecasting.  

Evaluated over 20 melt seasons (June–September, 2001–2020), IceDreamer outperforms 18 dynamical and statistical baselines. It achieves 9.53% RMSE and 96.89% ACC at a 6-month lead time, and uncovers a semiannual oscillation in sea-ice predictability aligned with lunar–solar cycles—revealing an astronomical modulation of Arctic climate variability .


### 2. Key Points
- The first dynamic Arctic Sea-Ice Prompt (SIP) corpus, encoding thermodynamics, cycles, and environmental conditions for physical knowledge injection .
- A novel Dream Learning framework that fuses textual prompts with observational data to improve seasonal pan-Arctic sea ice forecasts .
- A frozen Large Language Model (LLM) enables the model to learn physics-aware patterns and uncover measurable seasonality linked to astronomical cycles .


### 3. Framework Overview


IceDreamer’s architecture integrates three core modules to achieve physics-aware, multimodal sea ice forecasting:  
1. **Text-Guided Multi-Expert Fusion Module (T-MEFM)**: Aligns and fuses textual prompt features (encoded by LLMs) with sea ice concentration (SIC) and environmental data.  
2. **Dream Weaver Entangler (DWE)**: Entangles real observational features with synthetic "dream" features to expand the effective training manifold.  
3. **Dream Generator**: Produces high-fidelity synthetic samples during the "dream phase" to enhance model robustness .



### 4. Data & Code Availability
- **Observation Data**: 
  - NOAA/NSIDC Passive Microwave SIC Climate Data Record (Version 4, 1979–2022, 25 km resolution) .  
  - ERA5 reanalysis data (1979–2022, 0.25° resolution) and ORAS5 reanalysis data (1958–present) .  

- **SIP Corpus**:  
  - Dynamic textual prompt corpus: `SIP.txt` (contains temporally aligned physical, temporal, and environmental prompts) .  

- **Code Files**:  
  - Training script: `train Model/IceDreamer_train.py` (implements Dream Learning cycle, T-MEFM, and DWE training) .  
  - Testing script: `train Model/IceDreamer_test.py` (includes evaluation metrics calculation and forecast visualization) .  

