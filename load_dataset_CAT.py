from datasets import load_dataset
import pandas as pd

dataset = load_dataset('json', data_files='NekoQA-10K.json')

# str(dataset)
print(dataset)

# DatasetDict({
#     train: Dataset({
#         features: ['instruction', 'input', 'output', 'task_type', 'domain', 'metadata', 'answer_from', 'human_verified', 'copyright'],
#         num_rows: 240
#     })
# })


train_df = pd.DataFrame(dataset['train'])

# 查看 DataFrame
df = train_df[['instruction', 'output']]
df.head()
# print(df.head())
# print(df)