# 자료 / 2026-09-28 확인

- Gneiting, T. & Raftery, A. E. (2007). Strictly Proper Scoring Rules, Prediction, and Estimation. JASA 102(477), 359–378. DOI: https://doi.org/10.1198/016214506000001437
- Guo, C., Pleiss, G., Sun, Y. & Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. ICML/PMLR70:1321–1330. https://proceedings.mlr.press/v70/guo17a.html
- Angelopoulos, A. N. & Bates, S. A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification. https://arxiv.org/abs/2107.07511
- Warner et al. ModernBERT. https://arxiv.org/abs/2412.13663
- JHU CLSP mmBERT-base model card/config. https://huggingface.co/jhu-clsp/mmBERT-base ; https://huggingface.co/jhu-clsp/mmBERT-base/raw/main/config.json
- JHU CLSP mmBERT-small. https://huggingface.co/jhu-clsp/mmBERT-small
- Transformers ModernBERT reference mathematics (v4.57.1). https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/modernbert/modeling_modernbert.py
- SupersonicLabs Julia-1 official model card, interface, evaluation and limitations. https://huggingface.co/SupersonicLabs/Julia-1
- Hugging Face Tokenizers: Rust implementation and tokenizer.json. https://huggingface.co/docs/tokenizers/index
- SNLI. https://nlp.stanford.edu/projects/snli/ ; https://huggingface.co/datasets/stanfordnlp/snli
- MultiNLI. https://cims.nyu.edu/~sbowman/multinli/ ; https://huggingface.co/datasets/nyu-mll/multi_nli
- KLUE. https://github.com/KLUE-benchmark/KLUE ; https://huggingface.co/datasets/klue/klue
- Salesforce xLAM function-calling-60k. https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k

학습 목적함수의 결합, group-max conformal 적용, 공유 문맥 헤드와 네이티브 저장 형식은 이 코드의 설계입니다. 참고 논문의 실험 수치를 본 모델의 결과로 가져오지 않았습니다. 가중치 가져오기와 실제 데이터 어댑터는 공개 자료의 스키마를 바탕으로 작성했지만 제작 환경의 다운로드 제한 때문에 live 검증하지 못했습니다.
