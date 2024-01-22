### Transfer learning models for predicting leaf traits.
* **The number of the file represents the experimental steps in the paper.**
  
* **1_LUT_construction.py:** Generate the synthetic data using radiative transfer models (RTMs, PROSPECT-D and Leaf-SIP).
* **2_pretrain_DNN.py:** Pre-train the synthetic data using DNN models.
* **3_pretrained_DNN_to_obs.py:** Apply the pre-trained DNN model to the observational data directly.
* **4_partial_obs_GPR_PLSR_DNN.py:** Randomly Utilize partial observational data (10%, 20%, 30%,..., 80%) to train GPR, PLSR and DNN models.
* **5_fine_tune_DNN.py:** Randomly utilize partial observational data (10%, 20%, 30%,..., 80%) to fine-tune the pre-trianed DNN models.
* **6_GPR_PLSR_DNN_spatial_CV.py:** Leave one site out strategy to train GPR, PLSR and DNN models.
* **7_spatial_fine_tune.py:** Leave one site out strategy to fine-tune the pre-trianed DNN models.
* **8_GPR_PLSR_DNN_PFTs_CV.py:** Leave one PFT out strategy to train GPR, PLSR and DNN models.
* **9_PFTs_fine_tune.py:** Leave one site PFT strategy to fine-tune the pre-trianed DNN models.
* **10_GPR_PLSR_DNN_temporal_CV.py:** Leave one season out strategy to train GPR, PLSR and DNN models.
* **11_temporal_fine_tune.py:** Leave one season strategy to fine-tune the pre-trianed DNN models.
* **12_pure_PROSPECT_estimation.py:** Leaf trait estimation using pure PROSPECT-D model.
* **13_pure_LeafSIP_estimation.py:** Leaf trait estimation using pure leaf-SIP model.
* **Models.py:** Include the differnt modeling methods.
* **LeafSIP.py:** The code for Leaf-SIP model.
* **prospect_d.py:** The code for PROSPECT-D model.
* **spectral_library.py and dataSpec_PDB.csv:** The spectral library used in PROSPECT and Leaf-SIP, respectively.
* **0_saved_ML_model:** This folder contains the pre-trained DNN models and differnet trait models generated from each steps.
