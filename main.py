import torch, json, click, os
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

from data import all_context_from_corpus

MAX_CONTEXT_LENGTH = 512

def embed_batch(device, model, config, token_tensor):
    context_length = config['context_length']
    target_index = int(context_length / 2)

    segment_tensor = torch.ones_like(token_tensor)
    
    with torch.no_grad():
        segment_tensor = segment_tensor.to(device)
        token_tensor = token_tensor.to(device)
        output = model(token_tensor, segment_tensor)

    hidden_values = output[2]
    second_last_layer = hidden_values[-2]

    return second_last_layer[..., target_index, :]

def run_inference(device, config, batches):
    model = BertModel.from_pretrained(config['model_name'], output_hidden_states=True).to(device)
    model.eval()

    results_list = [embed_batch(device, model, config, batch).to('cpu') for batch in tqdm(batches, leave=True, desc="inference")]
    results = torch.cat(results_list, dim=0)

    return results


def save_results(results, metadata, word):
    if not os.path.exists('cache'): 
        os.makedirs('cache')

    with open(os.path.join('cache', f'{word}.pt'), 'wb') as FILE:
        torch.save((results, metadata), FILE)


@click.command()
@click.option('-w', '--word', help = 'Choice of word to plot.')
@click.option('-n', '--numfiles', help = 'The number of books to search.', default = 100)
def main(word, numfiles):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open("config.json", "r") as FILE:
        config = json.load(FILE)

    tokenizer = BertTokenizer.from_pretrained(config['model_name'])
    assert word in tokenizer.vocab.keys()

    batches, metadata = all_context_from_corpus(tokenizer, config, word, numfiles)
    results = run_inference(device, config, batches)
    save_results(results, metadata, word)


if __name__ == '__main__':
    main()

