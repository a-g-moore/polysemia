import torch, os
from joblib import Parallel, delayed
import contextlib, joblib
from tqdm import tqdm
import pandas as pd

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument
    
    Borrowed from somewhere on the internet. Thank you!
    """
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

def get_word_context_tensor(tokenizer, token_list, index, context_length):
    start_index = index - int(context_length / 2) + 1
    end_index = index + int(context_length / 2) - 1
    
    if start_index < 0 or end_index >= len(token_list):
        return None

    # Machine-readable context used for inferenece
    context_tokens = ["[CLS]"] + token_list[start_index:end_index] + ["[SEP]"]
    context_tensor = tokenizer.convert_tokens_to_ids(context_tokens)

    # Human-readable context for hover tooltips
    meta_context_tokens = token_list[(index-5):(index+6)]
    meta_context_tokens[5] = '<i>' + meta_context_tokens[5] + '</i>'
    meta_context = '...' + ' '.join(meta_context_tokens) + '...'
    meta_context = meta_context.replace(' ##', '')
    meta_context = meta_context.replace('##', '')

    return dict(
        tensor = torch.tensor(context_tensor), 
        context_string = meta_context
        )


def extract_context_tensors(tokenizer, filename, target_word, context_length):
    """
    Given a target word and a sequence of tokens in a file, this will return
    a PyTorch tensor with the context surrounding the word ready for 
    processing by BERT model (padded with [CLS] and [SEP]). Returns shape
    (num_occurances, context_length)
    """

    assert target_word in tokenizer.vocab.keys()

    # The data has each word on a new line. We want to put in in space-separated
    # format so that the tokenizer will be able to read it.
    try:
        with open(filename, 'r') as FILE:
            raw_text = ' '.join(FILE.readlines())
    except FileNotFoundError:
        print(f'File {filename} is listed in metadata but does not exist.')
        return None

    token_list = tokenizer.tokenize(raw_text)
    
    # Find word matches and get the context for each, filter out null responses
    context_pair_list = [get_word_context_tensor(tokenizer, token_list, index, context_length) 
            for index, word in enumerate(token_list) if word == target_word]
    context_tensor_list = [pair['tensor'].unsqueeze(0) for pair in context_pair_list if pair is not None]
    meta_context_list = [pair['context_string'] for pair in context_pair_list if pair is not None]

    if not len(context_tensor_list):
        return None

    block_tensor = torch.cat(context_tensor_list, dim=0)

    return dict(
        tensor = block_tensor,
        meta_context_list = meta_context_list,
        filename = filename
        )


def all_context_from_corpus(tokenizer, config, target_word, max_num_files):
    context_length = config['context_length']
    batch_size = config['batch_size']

    # Reformat paths as absolute
    current_master_dir = os.path.dirname(__file__)
    data_dir = os.path.join(current_master_dir, config['data_dir'])
    metadata_file = os.path.join(current_master_dir, config['metadata'])

    # Load metadata and process table
    table = pd.read_csv(metadata_file)
    table = table[table['language'] == "['en']"]
    table['id'] = table['id'].apply(lambda id : os.path.join(data_dir, id + "_tokens.txt"))
    existing_files = [os.path.join(data_dir, filename) for filename in os.listdir(data_dir)]
    table = table.query(f'id in {existing_files}')
    
    # Start a new process to load each individual file. 
    max_num_files = min(max_num_files, len(table))
    table = table.iloc[:max_num_files]
    with tqdm_joblib(tqdm(desc = "loading data", total = max_num_files)) as progress_bar:
        dicts = Parallel(n_jobs = -1)(
                delayed(extract_context_tensors)(tokenizer, os.path.join(data_dir, filename), target_word, context_length) 
                for filename in table['id'])
        table['context_dict'] = dicts

    # Filter out null results. 
    table = table.dropna(axis = 0, subset = 'context_dict')
    assert len(table)

    # Unpack all the tensors for each file and repackage them into uniform batches
    context_tensors = [batch['tensor'] for batch in table['context_dict']]
    single_batch = torch.cat(context_tensors, dim=0)
    packaged_batches = torch.split(single_batch, batch_size, dim=0)

    # Reformat metadata table to include a separate row for each instance of the given
    # word, with the book metadata replicated for all hits.
    table['context'] = table['context_dict'].apply(lambda batch : batch['meta_context_list'])
    table = table.drop(['context_dict'], axis=1)
    table = table.explode('context')
    
    return (packaged_batches, table)
