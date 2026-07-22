import torch
from torch import nn


class CheckinEmbedding(nn.Module):
    def __init__(
        self,
        embed_size,
        fusion_type,
        dataset_args
    ):
        super(CheckinEmbedding, self).__init__()
        self.embed_size = embed_size
        self.fusion_type = fusion_type
        num_user = max(dataset_args.num_user + 50, dataset_args.padding_user_id + 10)
        num_poi = max(dataset_args.num_poi + 50, dataset_args.padding_poi_id + 10)
        num_category = max(dataset_args.num_category + 50, dataset_args.padding_poi_category + 10)
        num_weekday = max(16, dataset_args.padding_weekday_id + 10)
        num_hour = max(32, dataset_args.padding_hour_id + 10)

        self.user_embedding = nn.Embedding(
            num_user,
            self.embed_size,
            padding_idx=dataset_args.padding_user_id if 0 <= dataset_args.padding_user_id < num_user else None
        )
        self.poi_embedding = nn.Embedding(
            num_poi,
            self.embed_size,
            padding_idx=dataset_args.padding_poi_id if 0 <= dataset_args.padding_poi_id < num_poi else None
        )
        self.category_embedding = nn.Embedding(
            num_category,
            self.embed_size,
            padding_idx=dataset_args.padding_poi_category if 0 <= dataset_args.padding_poi_category < num_category else None
        )
        self.dayofweek_embedding = nn.Embedding(
            num_weekday,
            self.embed_size,
            padding_idx=dataset_args.padding_weekday_id if 0 <= dataset_args.padding_weekday_id < num_weekday else None
        )
        self.hourofday_embedding = nn.Embedding(
            num_hour,
            self.embed_size,
            padding_idx=dataset_args.padding_hour_id if 0 <= dataset_args.padding_hour_id < num_hour else None
        )
        if self.fusion_type == 'concat':
            self.output_embed_size = 5 * self.embed_size
        elif self.fusion_type == 'add':
            self.output_embed_size = embed_size
        else:
            raise ValueError(f"Get wrong fusion type {self.fusion_type}")

    def forward(self, data):
        embedding_list = [
            self.user_embedding(data[..., 0].long()),
            self.poi_embedding(data[..., 1].long()),
            self.category_embedding(data[..., 2].long()),
            self.dayofweek_embedding(data[..., 6].long()),
            self.hourofday_embedding(data[..., 7].long())
        ]
        if self.fusion_type == 'concat':
            self.output_embed_size = len(embedding_list) * self.embed_size
            return torch.cat(embedding_list, -1)
        elif self.fusion_type == 'add':
            return torch.squeeze(sum(embedding_list))
        else:
            raise ValueError(f"Get wrong fusion type {self.fusion_type}")


class EdgeEmbedding(torch.nn.Module):
    def __init__(self, embed_size, fusion_type, num_edge_type):
        super(EdgeEmbedding, self).__init__()
        self.embed_size = embed_size
        self.fusion_type = fusion_type
        self.edge_type_embedding = nn.Embedding(num_edge_type, self.embed_size)
        self.output_embed_size = self.embed_size

    def forward(self, data):
        embedding_list = [self.edge_type_embedding(data.long())]

        if self.fusion_type == 'concat':
            self.output_embed_size = len(embedding_list) * self.embed_size
            return torch.cat(embedding_list, -1)
        elif self.fusion_type == 'add':
            return sum(embedding_list)
        else:
            raise ValueError(f"Get wrong fusion type {self.fusion_type}")
