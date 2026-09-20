"""Short reconstruction keeps target text out of the causal prompt."""
from pathlib import Path
import runpy


def test_shortening_versions_source_and_targets_the_complete_passage():
    shorten = runpy.run_path(str(Path(__file__).parents[1] /
                              'scripts/prepare_short_reconstruction.py'))['shorten_source']

    class Tokenizer:
        eos_token_id = 1

        def encode(self, text, *, add_special_tokens):
            assert not add_special_tokens
            return text.split()

    row = {'record_id': 'original', 'text': 'Title: Example\nPassage: ' +
           ' '.join(f'word{i}' for i in range(40)), 'created_at': 1,
           'kind': 'passage', 'provenance': {'context_sha256': 'hash',
                                          'article_title': 'Example'}}
    source, episode = shorten(row, Tokenizer(), words=32)
    assert source.record_id != row['record_id']
    assert source.text.endswith('word31')
    assert episode.answer == ' '.join(f'word{i}' for i in range(32))
    assert episode.answer not in episode.query
    assert episode.required_ids == (source.record_id,)
    assert episode.provenance['parent_source_id'] == row['record_id']
    assert shorten(row | {'text': 'Title: Example\nPassage: too short'}, Tokenizer()) is None
