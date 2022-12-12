class TrainingArguments(TrainingArguments):
    def __init__(self, *args, max_length=384, doc_stride=128, version_2_with_negative=False,
                 null_score_diff_threshold=0., n_best_size=20,  **kwargs):
        super().__init__(*args, **kwargs)

        self.max_length = max_length
        self.doc_stride = doc_stride
        self.version_2_with_negative = version_2_with_negative
        self.null_score_diff_threshold = null_score_diff_threshold
        self.n_best_size = n_best_size
        self.disable_tqdm = False
