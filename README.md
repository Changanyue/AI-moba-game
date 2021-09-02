# AI-moba-game
AI for moba game, trained on transformer  AI MOBA游戏，多模态预训练，包括王者荣耀，平安京，非人学园等等

使用faster-rcnn得到图片的region embedding，并把region的loc信息作为位置编码

预训练过程中，训练任务为：

1.mask图片的某个region和mask文本的token（不同时mask）

2.收集up主的视频，让模型做图文匹配

模型采用ViLBert架构，图片和文本进行attention交互
