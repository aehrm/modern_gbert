#%%
import os
from pathlib import Path

from streaming import MDSWriter
from torch.utils.data import DataLoader
from datasets import load_dataset
import more_itertools

from nltk.tokenize.punkt import PunktTokenizer
from tqdm import tqdm

#%%

data_files = list(str(x) for x in Path('/data/42-julia-hpc-rz-lsx/juw57zv/raw').glob('*head*'))
total_file_size = sum(os.path.getsize(x) for x in data_files)

jsonl_dataset = load_dataset('json', split='train', data_files=data_files, streaming=True)

#%%

def get_sentences(text, tokenizer):
    span_generator = tokenizer.span_tokenize(text, realign_boundaries=True)
    start, prev_start = None, None
    try:
        while True:
            start, _ = next(span_generator)
            if prev_start is not None:
                yield text[prev_start:start]

            prev_start = start
    except StopIteration:
        pass

    yield text[start:]


def split_text(batch, tokenizer):
    output_texts = []
    output_ids = []
    for text, sample_id in zip(batch['text'], batch['id']):
        start_idx = 0
        for sent_batch in more_itertools.constrained_batches(get_sentences(text, tokenizer), max_size=5000, strict=False):
            sent_batch = ''.join(sent_batch)
            output_texts.append(sent_batch)
            output_ids.append(sample_id+':'+str(start_idx))
            start_idx += len(sent_batch)
        # for i in range(0, len(text), 1024):
        #     output_texts.append(text[i:i+1024])
        #     output_ids.append(sample_id+':'+str(i))

    return {'text': output_texts, 'id': output_ids}


tokenizer = PunktTokenizer(lang='german')
prepared_dataset = jsonl_dataset.map(split_text, batched=True, batch_size=512, remove_columns=['source'], fn_kwargs=dict(tokenizer=tokenizer))

#%%

num_workers = min(64, jsonl_dataset.n_shards)
print(f'running with {num_workers=}')

columns = {'text': 'str', 'id': 'str'}
dl = DataLoader(prepared_dataset, num_workers=num_workers, batch_size=512)

def generate_samples(loader):
    for batch in loader:
        keys = list(batch.keys())
        current_bs = len(batch[keys[0]])
        for idx in range(current_bs):
            yield {k: v[idx].encode('utf-8') for k, v in batch.items()}

generator = generate_samples(dl)

with MDSWriter(columns=columns, out='/data/42-julia-hpc-rz-computerphil/ane53vq/llammlein_mds', exist_ok=True) as out:
    with tqdm(total=total_file_size, unit='B', unit_scale=True) as pbar:
        for sample in generator:
            out.write(sample)
            pbar.update(sum(len(x) for x in sample.values()))
