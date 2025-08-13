# MAP - Charting Student Math Misunderstandings

MAP——绘制学生数学误解图表

URL:https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/leaderboard

# debertav3large_0: 
  
`  `  PUBLIC SCORE = 0.54

`  `  RUN TIME = 4h on NVIDIA GeForce RTX 4060 for training + 2min on Kaggle GPU T4 x2 for inference

`  `  MODEL = microsoft/deberta-v3-large

`  `  Situation:训练几乎没有长进

# debertav3large_1: 
  
`  `  PUBLIC SCORE = 0.935

`  `  RUN TIME = 4h on NVIDIA GeForce RTX 4060 for training + 2min on Kaggle GPU T4 x2 for inference

`  `  MODEL = microsoft/deberta-v3-large

`  `  Situation:相对于0,降低了学习率并加入了lr_scheduler_type

# Baseline2: 
  
`  `  PUBLIC SCORE = 0.936

`  `  RUN TIME = 1m50s on GPU T4 x2 (Contains only the inference part)

`  `  MODEL = ModernBERT-Large-CV938

# Raw Deberta-v3 Large
  
`  `  PUBLIC SCORE = 0.002

`  `  RUN TIME = 1m48s on GPU T4 x2 (Contains only the inference part)

`  `  MODEL = Deberta-v3 Large

# Deberta-v3 easy train
  
`  `  PUBLIC SCORE = 0.927

`  `  RUN TIME = 3h on NVIDIA GeForce RTX 4060 for training + 1m50s on Kaggle GPU T4 x2 for inference

`  `  MODEL = Deberta-v3 Large
