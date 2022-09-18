import os
from time import perf_counter
from semen_analysis import prepare_data
from features import num_of_frame, extract_feature
import pandas as pd
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split



def load_and_extract(save_path=None):
    df = prepare_data()
    root_dir = "data/archive/VISEM_Tracking_Train_v4/Train/"
    df['n_sperm'] = pd.NA
    df['avg_dis'] = pd.NA
    df['avg_vec_dis'] = pd.NA
    df['avg_speed'] = pd.NA
    for row in df.iterrows():
        idx, data = row[0], row[1]
        video_id = row[1]['id']
        n_frame = num_of_frame(root_dir, video_id)

        # Extract feature
        n_name, avg_dis, avg_vector_distance, avg_speed = extract_feature(
            dir_path=root_dir, video_id=video_id, num_frame=n_frame)

        # Write to pandas
        data['n_sperm'] = n_name
        data['avg_dis'] = avg_dis
        data['avg_vec_dis'] = avg_vector_distance
        data['avg_speed'] = avg_speed
        row = (idx, data)
        df.iloc[idx] = data
        pass
    if type(save_path) is str:
        df.to_csv(save_path, sep=",", index=False)
    return df


class SimpleDataset(Dataset):

    def __init__(self,
                 save_path="train_feats.csv",
                 group='train',
                 non_split_path="feats.csv") -> None:
        super().__init__()
        assert group in ["train", "validation"], f"{group} not in enum!"

        if not os.path.exists(save_path):
            if not os.path.exists(non_split_path):
                self.df = load_and_extract(save_path=non_split_path)
            else:
                self.df = pd.read_csv(non_split_path, sep=",")
            train_df, val_df = train_test_split(self.df, test_size=0.4)
            train_df.to_csv(f"train_{non_split_path}", sep=",", index=False)
            val_df.to_csv(f"val_{non_split_path}", sep=",", index=False)
            if group == 'train': self.df = train_df
            else: self.df = val_df
        else:
            self.df = pd.read_csv(
                save_path,
                sep=",",
            )

    def __getitem__(self, index):
        data = self.df.iloc[index]
        progressive = data['percent_progressive'] / 100.
        non_progressive = data['percent_non_progressive'] / 100.
        immotile = data['percent_immotile'] / 100.
        n_sperm = data['n_sperm']
        avg_dis = data['avg_dis']
        avg_vec_dis = data['avg_vec_dis']
        avg_speed = data['avg_speed']
        # Make input
        _input = Tensor([n_sperm, avg_dis, avg_vec_dis, avg_speed])
        _label = Tensor([progressive, non_progressive, immotile])
        return dict(data=_input, label=_label)

    def __len__(self):
        return self.df.shape[0]


class SimpleModel(nn.Module):

    def __init__(self, n_in=4, n_out=3, is_softmax=True) -> None:
        super().__init__()
        self.linear = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 3))
        self.is_softmax = is_softmax

    def forward(self, x):
        output = self.linear(x)
        if self.is_softmax:
            return torch.nn.functional.softmax(output)
        return output


if __name__ == "__main__":
    # train_dataset = SimpleDataset(group="train", save_path="train_feats.csv")
    # val_dataset = SimpleDataset(group="validation", save_path="val_feats.csv")
    # train_loader = DataLoader(train_dataset,
    #                           batch_size=4,
    #                           shuffle=True,
    #                           drop_last=False)
    # val_loader = DataLoader(val_dataset,
    #                         batch_size=len(val_dataset),
    #                         shuffle=True,
    #                         drop_last=False)

    # model = SimpleModel()

    # learning_rate = 1e-4
    # optimizer = torch.optim.Adam(model.parameters(), learning_rate)

    # n_epoch = 1000
    # criterion = nn.L1Loss()
    # print(
    #     f"Train sample: {len(train_dataset)} / Val sample: {len(val_dataset)} "
    # )
    # train_losses = []
    # val_losses = []
    # for epoch in range(n_epoch):
    #     for batch in train_loader:
    #         pred = model(batch['data'])
    #         loss = criterion(pred, batch['label'])

    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         pass
    #     print(f"[{epoch:>5d}/{n_epoch:>5d}]\n" +
    #           f"├── train_loss    : {loss:>7f}")
    #     train_losses.append((epoch, float(loss)))

    #     with torch.no_grad():
    #         for batch in val_loader:
    #             pred = model(batch['data'])
    #             val_loss = criterion(pred, batch['label'])
    #             print(f"[{epoch:>5d}/{n_epoch:>5d}]\n" +
    #                   f"├── val_loss    : {val_loss:>7f}")
    #         val_losses.append((epoch, float(val_loss)))
    #     pass
    # torch.save(dict(train_losses=train_losses, val_losses=val_losses),
    #            "train.pt")
    pass