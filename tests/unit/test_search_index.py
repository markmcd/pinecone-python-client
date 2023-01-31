from core.grpc.search_index import SearchIndex, Document
import pandas as pd
import os


class TestSearchIndex:

    def setup_class(self):
        self.df_len = 1023
        self.df = pd.DataFrame({'id': [str(i) for i in range(self.df_len)],
                                'text': ['foo bar'] * self.df_len,
                                'foo': ["bar"] * self.df_len})

        cur_file_path = os.path.dirname(os.path.abspath(__file__))
        self.df_path = os.path.join(cur_file_path, 'test_data.csv')

        self.df.to_csv(self.df_path, index=False)

        self.jsonl_path = os.path.join(cur_file_path, 'test_data.jsonl')
        with open(self.jsonl_path, 'w') as f:
            for i in range(self.df_len):
                f.write(f'{{"id": "{i}", "text": "foo bar", "foo": "bar"}}\n')

    def teardown_class(self):
        if os.path.exists(self.df_path):
            os.remove(self.df_path)
        if os.path.exists(self.jsonl_path):
            os.remove(self.jsonl_path)

    def test_iterDocDataset_dataframeIntput_getIter(self):
        # creates a dataframe with 1000 rows
        chunks = list(SearchIndex._iter_doc_dataset(dataset=self.df,
                                                    text_col_name='text',
                                                    id_col_name='id',
                                                    metadata_col_names=['foo'],
                                                    batch_size=100))
        ids = (i for i in range(self.df_len))
        for i in range(10):
            assert chunks[i] == [Document(id=str(next(ids)), text='foo bar', metadata={'foo': 'bar'})
                                 for _ in range(100)]
        assert chunks[10] == [Document(id=str(next(ids)), text='foo bar', metadata={'foo': 'bar'})
                              for _ in range(23)]

    def test_iterDocDataset_csvIntput_getIter(self):
        chunks = list(SearchIndex._iter_doc_dataset(dataset=self.df_path,
                                                    text_col_name='text',
                                                    id_col_name='id',
                                                    metadata_col_names=['foo'],
                                                    batch_size=100))

        ids = (i for i in range(self.df_len))
        for i in range(10):
            assert chunks[i] == [Document(id=str(next(ids)), text='foo bar', metadata={'foo': 'bar'})
                                 for _ in range(100)]
        assert chunks[10] == [Document(id=str(next(ids)), text='foo bar', metadata={'foo': 'bar'})
                              for _ in range(23)]

    def test_iterDocDataset_jsonlIntput_getIter(self):
        chunks = list(SearchIndex._iter_doc_dataset(dataset=self.jsonl_path,
                                                    text_col_name='text',
                                                    id_col_name='id',
                                                    metadata_col_names=['foo'],
                                                    batch_size=100))

        ids = (i for i in range(self.df_len))
        for i in range(10):
            assert chunks[i] == [Document(id=str(next(ids)), text='foo bar', metadata={'foo': 'bar'})
                                 for _ in range(100)]
        assert chunks[10] == [Document(id=str(next(ids)), text='foo bar', metadata={'foo': 'bar'})
                              for _ in range(23)]

