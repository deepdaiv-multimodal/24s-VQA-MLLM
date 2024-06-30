import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from .utils.load_data import CommonData, DataSet
from .model.beit3 import BEiT3ForVisualQuestionAnswering
import deepspeed

class Runner:
    def __init__(self, __C, evaluator):
        self.__C = __C
        self.evaluator = evaluator

    def train(self, train_set, valid_set):
        net = BEiT3ForVisualQuestionAnswering(self.__C, num_classes=train_set.ans_size)

        # Load pretrained weights
        if os.path.exists(self.__C.PRETRAINED_MODEL_PATH):
            checkpoint = torch.load(self.__C.PRETRAINED_MODEL_PATH, map_location="cuda")
            net.load_state_dict(checkpoint, strict=False)
            print(f"Loaded pretrained weights from {self.__C.PRETRAINED_MODEL_PATH}")

        # 기본 AdamW 옵티마이저 설정
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, net.parameters()), 
            lr=self.__C.LR_BASE, 
            betas=self.__C.OPT_BETAS, 
            weight_decay=self.__C.WEIGHT_DECAY
        )

        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=net,
            optimizer=optimizer,
            config=self.__C.deepspeed_config
        )

        dataloader = DataLoader(train_set, batch_size=self.__C.BATCH_SIZE, shuffle=True, num_workers=2)

        for epoch in range(self.__C.EPOCHS):
            model_engine.train()
            for step, input_tuple in enumerate(dataloader):
                img, ques_ids, target = input_tuple

                img = img.cuda()
                ques_ids = ques_ids.cuda()
                target = target.cuda()

                optimizer.zero_grad()
                pred = model_engine(img, question=ques_ids)
                loss = torch.nn.functional.cross_entropy(pred, target)
                model_engine.backward(loss)
                model_engine.step()

                if step % 10 == 0:
                    print(f'Epoch {epoch}, Step {step}, Loss {loss.item()}')

    def run(self):
        common_data = CommonData(self.__C)
        train_set = DataSet(self.__C, common_data, self.__C.TRAIN_SPLITS)
        valid_set = DataSet(self.__C, common_data, self.__C.EVAL_SPLITS)
        self.train(train_set, valid_set)