import pandas as pd
from typing import Dict, Tuple
from sklearn.preprocessing import LabelEncoder
import logging


def id_encode(
        fit_df: pd.DataFrame,
        encode_df: pd.DataFrame,
        column: str,
        padding: int = -1
) -> Tuple[LabelEncoder, int]:
    id_le = LabelEncoder()
    id_le = id_le.fit(fit_df[column].values.tolist())

    classes_list = id_le.classes_.tolist()
    mapping = {val: idx for idx, val in enumerate(classes_list)}

    if padding == 0:
        padding_id = padding
        # 从 1 开始映射，缺失的填 0
        mapping_shifted = {val: idx + 1 for val, idx in mapping.items()}
        encode_df[column] = encode_df[column].map(mapping_shifted).fillna(padding_id).astype(int)
    else:
        padding_id = len(classes_list)
        # 从 0 开始映射，缺失的填 padding_id
        encode_df[column] = encode_df[column].map(mapping).fillna(padding_id).astype(int)

    return id_le, padding_id


def ignore_first(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ignore the first check-in sample of every trajectory because of no historical check-in.

    """
    df['pseudo_session_trajectory_rank'] = df.groupby(
        'pseudo_session_trajectory_id')['UTCTimeOffset'].rank(method='first')
    df['query_pseudo_session_trajectory_id'] = df['pseudo_session_trajectory_id'].shift()
    df.loc[df['pseudo_session_trajectory_rank'] == 1, 'query_pseudo_session_trajectory_id'] = None
    df['last_checkin_epoch_time'] = df['UTCTimeOffsetEpoch'].shift()
    df.loc[df['pseudo_session_trajectory_rank'] == 1, 'last_checkin_epoch_time'] = None
    df.loc[df['UserRank'] == 1, 'SplitTag'] = 'ignore'
    df.loc[df['pseudo_session_trajectory_rank'] == 1, 'SplitTag'] = 'ignore'
    return df


def only_keep_last(df: pd.DataFrame) -> pd.DataFrame:
    """
    Only keep the last check-in samples in validation and testing for measuring model performance.

    """
    df['pseudo_session_trajectory_count'] = df.groupby(
        'pseudo_session_trajectory_id')['UTCTimeOffset'].transform('count')
    df.loc[(df['SplitTag'] == 'validation') & (
            df['pseudo_session_trajectory_count'] != df['pseudo_session_trajectory_rank']
    ), 'SplitTag'] = 'ignore'
    df.loc[(df['SplitTag'] == 'test') & (
            df['pseudo_session_trajectory_count'] != df['pseudo_session_trajectory_rank']
    ), 'SplitTag'] = 'ignore'
    return df


def remove_unseen_user_poi(df: pd.DataFrame) -> Dict:
    """
    Remove the samples of Validate and Test if those POIs or Users didnt show in training samples

    """
    preprocess_result = dict()
    df_train = df[df['SplitTag'] == 'train']
    df_validate = df[df['SplitTag'] == 'validation']
    df_test = df[df['SplitTag'] == 'test']

    train_user_set = set(df_train['UserId'])
    train_poi_set = set(df_train['PoiId'])
    df_validate = df_validate[
        (df_validate['UserId'].isin(train_user_set)) & (df_validate['PoiId'].isin(train_poi_set))].reset_index()
    df_test = df_test[(df_test['UserId'].isin(train_user_set)) & (df_test['PoiId'].isin(train_poi_set))].reset_index()

    preprocess_result['sample'] = df
    preprocess_result['train_sample'] = df_train
    preprocess_result['validate_sample'] = df_validate
    preprocess_result['test_sample'] = df_test

    logging.info(
        f"[Preprocess] train shape: {df_train.shape}, validation shape: {df_validate.shape}, "
        f"test shape: {df_test.shape}"
    )
    return preprocess_result
