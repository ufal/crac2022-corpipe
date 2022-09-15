#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import datetime
import functools
import json
import os
import pickle
import shutil
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import transformers

import udapi
import udapi.block.corefud.movehead
import udapi.block.corefud.removemisc

parser = argparse.ArgumentParser()
parser.add_argument("data", default=[], nargs="+", type=str, help="Data.")
parser.add_argument("--batch_size", default=8, type=int, help="Batch size.")
parser.add_argument("--bert", default="bert-base-multilingual-cased", type=str, help="Bert model.")
parser.add_argument("--beta_2", default=0.999, type=float, help="Beta2.")
parser.add_argument("--crf", default=False, action="store_true", help="Use CRF.")
parser.add_argument("--debug", default=False, action="store_true", help="Debug mode.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--exp", default="", type=str, help="Exp name.")
parser.add_argument("--label_smoothing", default=0.1, type=float, help="Label smoothing.")
parser.add_argument("--lazy_adam", default=False, action="store_true", help="Use Lazy Adam.")
parser.add_argument("--learning_rate", default=1e-5, type=float, help="Learning rate.")
parser.add_argument("--learning_rate_decay", default=False, action="store_true", help="Decay LR.")
parser.add_argument("--max_links", default=None, type=int, help="Max antecedent links to train on.")
parser.add_argument("--resample", default=[], nargs="*", type=float, help="Train data resample ratio.")
parser.add_argument("--right", default=None, type=int, help="Reserved space for right context, if any.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--segment", default=256, type=int, help="Segment size")
parser.add_argument("--threads", default=4, type=int, help="Maximum number of threads to use.")
parser.add_argument("--train", default=[], nargs="+", type=str, help="Additional train data.")
parser.add_argument("--treebank_id", default=False, action="store_true", help="Use treebank id.")
parser.add_argument("--warmup", default=0.1, type=float, help="Warmup ratio.")

class Dataset:
    TOKEN_EMPTY = "\\"
    # Use Katakana as treebank ids -- use a subset with embedded "space" characters in Rembert
    TOKEN_TREEBANKS = [chr(i) for i in [*range(0x30A2, 0x30AB, 2), *range(0x30AB, 0x30B0, 2), *range(0x30B3, 0x30BC, 2)]]

    def __init__(self, path: str, tokenizer: transformers.PreTrainedTokenizerFast, treebank_id: int) -> None:
        self._cls = tokenizer.cls_token_id
        self._sep = tokenizer.sep_token_id
        self._path = path
        self._treebank_token = []
        if treebank_id: # 0 is deliberately considered as no treebank id
            self._treebank_token = tokenizer.encode(self.TOKEN_TREEBANKS[treebank_id - 1], add_special_tokens=False)
            if self._treebank_token[0] == 6 and "xlm-r" in tokenizer.name_or_path: # Hack: remove the space-only token in XLM-R
                self._treebank_token = self._treebank_token[1:]
            assert len(self._treebank_token) == 1

        # Create the tokenized documents if they do not exist
        cache_path = f"{path}.mentions.{os.path.basename(tokenizer.name_or_path)}"
        if not os.path.exists(cache_path):
            # Create flat representation
            if not os.path.exists(f"{path}.flat"):
                with open(path, "r", encoding="utf-8-sig") as data_file:
                    data_original = [line.rstrip("\r\n") for line in data_file.readlines() if not re.match(r"^\d+-", line)]

                # Remove multi-word tokens
                data = [line for line in data_original if not re.match(r"\d+-", line)]

                # Flatten the representation
                flat, i = [], 0
                for line in data:
                    if not line:
                        i = 0
                    elif not line.startswith("#"):
                        columns = line.split("\t")
                        assert len(columns) == 10
                        if "." in columns[0]:
                            columns[1] = self.TOKEN_EMPTY + " " + (columns[1] if columns[1] and columns[1] != "_" else columns[2])
                        columns[0] = str(i + 1)
                        columns[6] = "0"
                        line = "\t".join(columns)
                        i += 1
                    flat.append(line)

                with open(f"{path}.flat", "w", encoding="utf-8-sig") as data_file:
                    for line in flat:
                        print(line, file=data_file)

            # Parse with Udapi
            if not os.path.exists(f"{path}.mentions"):
                docs = []
                for doc in udapi.block.read.conllu.Conllu(files=[f"{path}.flat"], split_docs=True).read_documents():
                    new_doc = []
                    for tree in doc.trees:
                        words, coref_mentions = [], set()
                        for node in tree.descendants:
                            words.append(node.form)
                            coref_mentions.update(node.coref_mentions)

                        dense_mentions = []
                        for mention in coref_mentions:
                            span = mention.words
                            start = end = span.index(mention.head)
                            while start > 0 and span[start - 1].ord + 1 == span[start].ord: start -= 1
                            while end < len(span) - 1 and span[end].ord + 1== span[end + 1].ord: end += 1
                            dense_mentions.append(((span[start].ord - 1, span[end].ord - 1), mention.entity.eid, start > 0 or end + 1 < len(span)))
                        dense_mentions = sorted(dense_mentions, key=lambda x:(x[0][0], -x[0][1], x[2]))

                        mentions = []
                        for i, mention in enumerate(dense_mentions):
                            if i and dense_mentions[i-1][0] == mention[0]:
                                print(f"Multiple same mentions {mention[2]}/{dense_mentions[i-1][2]} in sent_id {tree.sent_id}: {tree.get_sentence()}", flush=True)
                                continue
                            mentions.append((mention[0][0], mention[0][1], mention[1]))
                        new_doc.append((words, mentions))
                    docs.append(new_doc)
                with open(f"{path}.mentions", "wb") as cache_file:
                    pickle.dump(docs, cache_file, protocol=3)
            with open(f"{path}.mentions", "rb") as cache_file:
                docs = pickle.load(cache_file)

            # Tokenize the data, generate stack operations and subword mentions
            self.docs = []
            for doc in docs:
                new_doc = []
                for words, mentions in doc:
                    subwords, word_indices, word_tags, subword_mentions, stack = [], [], [], [], []
                    for i in range(len(words)):
                        word_indices.append(len(subwords))
                        word = (" " if "robeczech" in tokenizer.name_or_path else "") + words[i]
                        subword = tokenizer.encode(word, add_special_tokens=False)
                        assert len(subword) > 0
                        if subword[0] == 6 and "xlm-r" in tokenizer.name_or_path: # Hack: remove the space-only token in XLM-R
                            subword = subword[1:]
                        assert len(subword) > 0
                        subwords.extend(subword)

                        tag = [str(len(stack))]
                        for _ in range(2):
                            for j in reversed(range(len(stack))):
                                start, end, eid = stack[j]
                                if end == i:
                                    tag.append(f"POP:{len(stack)-j}")
                                    subword_mentions.append((start, word_indices[-1], eid))
                                    stack.pop(j)
                            while mentions and mentions[0][0] == i:
                                tag.append("PUSH")
                                stack.append((word_indices[-1], mentions[0][1], mentions[0][2]))
                                mentions = mentions[1:]
                        word_tags.append(",".join(tag))
                    assert len(stack) == 0
                    subword_mentions = sorted(subword_mentions, key=lambda x:(x[0], -x[1]))

                    new_doc.append((subwords, word_indices, word_tags, subword_mentions))
                self.docs.append(new_doc)

            with open(cache_path, "wb") as cache_file:
                pickle.dump(self.docs, cache_file, protocol=3)
        with open(cache_path, "rb") as cache_file:
            self.docs = pickle.load(cache_file)

    @staticmethod
    def create_tags(trains: list[Dataset]) -> list[str]:
        tags = set()
        for train in trains:
            for doc in train.docs:
                for _, _, word_tags, _ in doc:
                    tags.update(word_tags)
        return sorted(tags)

    def pipeline(self, tags_map: dict[str, int], train: bool, args: argparse.Namespace) -> tf.data.Dataset:
        def generator():
            tid = len(self._treebank_token)
            for doc in self.docs:
                p_subwords, p_subword_mentions = [], []
                for doc_i, (subwords, word_indices, word_tags, subword_mentions) in enumerate(doc):
                    assert train or len(subwords) + 4 + tid <= args.segment
                    if len(subwords) + 4 + tid <= args.segment:
                        right_reserve = min((args.segment - 4 - tid - len(subwords)) // 2, args.right or 0)
                        context = min(args.segment - 4 - tid - len(subwords) - right_reserve, len(p_subwords))
                        word_indices = [context + 2 + tid + i for i in word_indices + [len(subwords)]]
                        e_subwords = [self._cls, *self._treebank_token, *p_subwords[-context:], self._sep, *subwords, self._sep]
                        if args.right is not None:
                            i = doc_i + 1
                            while i < len(doc) and len(e_subwords) + 1 < args.segment:
                                e_subwords.extend(doc[i][0][:args.segment - len(e_subwords) - 1])
                                i += 1
                        e_subwords.append(self._sep)

                        output = (e_subwords, word_indices)
                        if train:
                            offset = len(p_subwords) - context
                            prev = [(s - offset + 1 + tid, e - offset + 1 + tid, eid) for s, e, eid in p_subword_mentions if s >= offset]
                            prev_pos = np.array([[s, e] for s, e, _ in prev], dtype=np.int32).reshape([-1, 2])
                            prev_eid = np.array([eid for _, _, eid in prev], dtype=str)
                            ment = [(context + 2 + tid + s, context + 2 + tid + e, eid) for s, e, eid in subword_mentions]
                            ment_pos = np.array([[s, e] for s, e, _ in ment], dtype=np.int32).reshape([-1, 2])
                            ment_eid = np.array([eid for _, _, eid in ment], dtype=str)
                            mask = ment_pos[:, 0, None] > np.concatenate([prev_pos[:, 0], ment_pos[:, 0]])[None, :]
                            diag = np.pad(np.eye(len(ment_pos)), [[0, 0], [len(prev_pos), 0]])
                            gold = (ment_eid[:, None] == np.concatenate([prev_eid, ment_eid])[None, :]) * mask
                            gold = np.where(np.sum(gold, axis=1, keepdims=True) > 0, gold, diag)
                            if args.max_links is not None:
                                max_link_mask = np.cumsum(gold, axis=1)
                                gold *= (max_link_mask > max_link_mask[:, -1:] - args.max_links)
                            gold /= np.sum(gold, axis=1, keepdims=True)
                            mask = mask + diag
                            if args.label_smoothing:
                                gold = (1 - args.label_smoothing) * gold + args.label_smoothing * (mask / np.sum(mask, axis=1, keepdims=True))

                            word_tags = [tags_map[tag] for tag in word_tags]
                            output = (output, (word_tags, prev_pos, ment_pos, mask, gold))
                        yield output

                    p_subword_mentions.extend((s + len(p_subwords), e + len(p_subwords), eid) for s, e, eid in subword_mentions)
                    p_subwords.extend(subwords)

        output_signature=(tf.TensorSpec([None], tf.int32), tf.TensorSpec([None], tf.int32))
        if train:
            output_signature = (output_signature, (
                tf.TensorSpec([None], tf.int32), tf.TensorSpec([None, 2], tf.int32), tf.TensorSpec([None, 2], tf.int32),
                tf.TensorSpec([None, None], tf.bool), tf.TensorSpec([None, None], tf.float32),
            ))

        pipeline = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
        pipeline = pipeline.cache()
        pipeline = pipeline.apply(tf.data.experimental.assert_cardinality(sum(1 for _ in pipeline)))
        return pipeline

    def save_mentions(self, path: str, headsonly_path: str, mentions: list[list[tuple[int, int, int]]]) -> None:
        doc = udapi.block.read.conllu.Conllu(files=[self._path]).read_documents()[0]
        udapi.block.corefud.removemisc.RemoveMisc(attrnames="Entity,SplitAnte,Bridge").apply_on_document(doc)

        entities = {}
        for i, tree in enumerate(doc.trees):
            nodes = tree.descendants_and_empty
            for start, end, eid in mentions[i]:
                if not eid in entities:
                    entities[eid] = udapi.core.coref.CorefEntity(f"c{eid}")
                udapi.core.coref.CorefMention(nodes[start:end + 1], entity=entities[eid])
        doc._eid_to_entity = {entity._eid: entity for entity in sorted(entities.values())}
        udapi.block.corefud.movehead.MoveHead(bugs='ignore').apply_on_document(doc)
        udapi.block.write.conllu.Conllu(files=[path]).apply_on_document(doc)

        for mention in doc.coref_mentions:
            mention.words = [mention.head]
        udapi.block.write.conllu.Conllu(files=[headsonly_path]).apply_on_document(doc)


class Model(tf.keras.Model):
    def __init__(self, tags: list[str], args: argparse.Namespace) -> None:
        super().__init__()
        self._tags = tags
        self._args = args

        assert tags[0] == "0" # Used as a boundary tag in CRF
        self._boundary_logits = tf.cast(tf.range(len(tags)) > 0, tf.float32) * -1e6

        self._bert = transformers.TFAutoModel.from_pretrained(
            args.bert, from_pt=any(m in args.bert.lower() for m in ["rubert", "herbert", "flaubert", "litlat", "roberta-base-ca", "spanbert"]))
        self._dense_hidden_q = tf.keras.layers.Dense(4 * self._bert.config.hidden_size, activation=tf.nn.relu)
        self._dense_hidden_k = tf.keras.layers.Dense(4 * self._bert.config.hidden_size, activation=tf.nn.relu)
        self._dense_hidden_tags = tf.keras.layers.Dense(4 * self._bert.config.hidden_size, activation=tf.nn.relu)
        self._dense_q = tf.keras.layers.Dense(self._bert.config.hidden_size, use_bias=False)
        self._dense_k = tf.keras.layers.Dense(self._bert.config.hidden_size, use_bias=False)
        self._dense_tags = tf.keras.layers.Dense(len(tags))
        if args.crf:
            self._crf_weights = self.add_weight(
                name="crf_weights", shape=[len(tags), len(tags)], dtype=tf.float32, initializer=tf.initializers.Orthogonal())

    def compile(self, train: tf.data.Dataset) -> None:
        args = self._args
        warmup_steps = int(args.warmup * args.epochs * len(train))
        learning_rate = tf.optimizers.schedules.PolynomialDecay(
            args.learning_rate, args.epochs * len(train) - warmup_steps, 0. if args.learning_rate_decay else args.learning_rate)
        if warmup_steps:
            class LinearWarmup(tf.optimizers.schedules.LearningRateSchedule):
                def __init__(self, warmup_steps, following_schedule):
                    self._warmup_steps = warmup_steps
                    self._warmup = tf.optimizers.schedules.PolynomialDecay(0., warmup_steps, following_schedule(0))
                    self._following = following_schedule
                def __call__(self, step):
                    return tf.cond(step < self._warmup_steps,
                                   lambda: self._warmup(step),
                                   lambda: self._following(step - self._warmup_steps))
            learning_rate = LinearWarmup(warmup_steps, learning_rate)
        optimizer = tfa.optimizers.LazyAdam if args.lazy_adam else tf.optimizers.Adam
        super().compile(optimizer=optimizer(learning_rate=learning_rate, beta_2=args.beta_2))

    def crf_loss(self, gold_labels: tf.RaggedTensor, logits: tf.RaggedTensor) -> tf.Tensor:
        boundary_logits = tf.broadcast_to(self._boundary_logits, [logits.bounding_shape(0), 1, len(self._tags)])
        logits = tf.concat([boundary_logits, logits, boundary_logits], axis=1)
        boundary_labels = tf.zeros([logits.bounding_shape(0), 1], tf.int32)
        gold_labels = tf.concat([boundary_labels, gold_labels, boundary_labels], axis=1)
        loss, _ = tfa.text.crf_log_likelihood(logits.to_tensor(), gold_labels.to_tensor(), gold_labels.row_lengths(), self._crf_weights)
        return tf.math.reduce_sum(-loss) / tf.cast(tf.math.reduce_sum(gold_labels.row_lengths()), tf.float32)

    def crf_decode(self, logits: tf.RaggedTensor) -> tf.RaggedTensor:
        boundary_logits = tf.broadcast_to(self._boundary_logits, [logits.bounding_shape(0), 1, len(self._tags)])
        logits = tf.concat([boundary_logits, logits, boundary_logits], axis=1)
        predictions, _ = tfa.text.crf_decode(logits.to_tensor(), self._crf_weights, logits.row_lengths())
        predictions = tf.RaggedTensor.from_tensor(predictions, logits.row_lengths())
        predictions = predictions[:, 1:-1]
        return predictions

    @tf.function(experimental_relax_shapes=True)
    def compute_tags(self, subwords, word_indices, training) -> tuple[tf.RaggedTensor, tf.RaggedTensor]:
        embeddings = self._bert(subwords.to_tensor(), attention_mask=tf.sequence_mask(subwords.row_lengths()), training=training).last_hidden_state
        words = tf.gather(embeddings, word_indices[:, :-1], batch_dims=1)
        logits = self._dense_tags(self._dense_hidden_tags(words))
        return embeddings, logits

    @tf.function(experimental_relax_shapes=True)
    def compute_antecedents(self, embeddings, previous, mentions) -> tf.RaggedTensor:
        mentions_embedded = tf.gather(embeddings, mentions, batch_dims=1).values
        mentions_embedded = tf.reshape(mentions_embedded, [-1, np.prod(mentions_embedded.shape[-2:])])
        queries = mentions.with_values(self._dense_q(self._dense_hidden_q(mentions_embedded)))
        keys_mentions = mentions.with_values(self._dense_k(self._dense_hidden_k(mentions_embedded)))

        previous_embedded = tf.gather(embeddings, previous, batch_dims=1).values
        previous_embedded = tf.reshape(previous_embedded, [-1, mentions_embedded.shape[-1]])
        keys_previous = previous.with_values(self._dense_k(self._dense_hidden_k(previous_embedded)))
        keys = tf.concat([keys_previous, keys_mentions], axis=1)
        weights = tf.matmul(queries.to_tensor(), keys.to_tensor(), transpose_b=True) / (self._dense_q.units ** 0.5)
        return weights

    def train_step(self, data: tuple) -> dict[str, tf.Tensor]:
        (subwords, word_indices), (tags, previous, mentions, mask, antecedents) = data
        with tf.GradientTape() as tape:
            # Tagging part
            embeddings, logits = self.compute_tags(subwords, word_indices, True)
            if self._args.crf:
                tags_loss = self.crf_loss(tags, logits)
            else:
                tags_loss = tf.losses.CategoricalCrossentropy(
                    from_logits=True, label_smoothing=self._args.label_smoothing, reduction=tf.losses.Reduction.SUM)(
                        tf.one_hot(tags.values, len(self._tags)), logits.values) / tf.cast(tf.shape(logits.values)[0], tf.float32)

            # Antecedents part
            def antecedent_loss():
                weights = self.compute_antecedents(embeddings, previous, mentions)
                mask_dense = tf.cast(mask.to_tensor(), tf.float32)
                weights = weights[:, :, :tf.shape(mask_dense)[-1]] # Happens when the largest number of mentions have 0 queries
                weights = mask_dense * weights + (1 - mask_dense) * -1e9
                return tf.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.losses.Reduction.SUM)(
                    antecedents.values.to_tensor(), tf.RaggedTensor.from_tensor(weights, antecedents.row_lengths()).values
                ) / tf.cast(tf.math.reduce_sum(antecedents.row_lengths()), tf.float32)
            antecedent_loss = tf.cond(tf.math.reduce_sum(antecedents.row_lengths()) != 0, antecedent_loss, lambda: 0.)

            loss = tags_loss + antecedent_loss

        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        return {"tags_loss": tags_loss, "antecedent_loss": antecedent_loss, "loss": loss,
                "lr": self.optimizer.learning_rate(self.optimizer.iterations)}

    def predict(self, dataset: Dataset, pipeline: tf.data.Dataset) -> list[list[tuple[int, int, int]]]:
        tid = len(dataset._treebank_token)
        results, entities = [], 0
        doc_mentions, doc_subwords = [], 0
        for b_subwords, b_word_indices in pipeline:
            b_size = b_subwords.shape[0]
            b_embeddings, b_logits = self.compute_tags(b_subwords, b_word_indices, False)
            b_tags = self.crf_decode(b_logits) if self._args.crf else b_logits.with_values(tf.argmax(b_logits.values, axis=-1))

            b_previous, b_mentions, b_refs = [], [], []
            for b in range(b_size):
                word_indices, tags = b_word_indices[b].numpy(), b_tags[b].numpy()
                if word_indices[0] == 2 + tid:
                    doc_mentions, doc_subwords = [], 0

                # Decode mentions
                mentions, stack = [], []
                for i, tag in enumerate(self._tags[tag] for tag in tags):
                    for command in tag.split(",")[1:]: # The first is stack depth, which we ignore now
                        if command == "PUSH":
                            stack.append(i)
                        elif command.startswith("POP:"):
                            j = int(command.removeprefix("POP:"))
                            if len(stack):
                                j = len(stack) - (j if j <= len(stack) else 1)
                                mentions.append((stack.pop(j), i))
                        else:
                            raise ValueError(f"Unknown command '{command}'")
                while len(stack):
                    mentions.append((stack.pop(), len(tags) - 1))
                mentions = [[s, e, None] for s, e in sorted(set(mentions), key=lambda x: (x[0], -x[1]))]

                # Prepare inputs for antecedent prediction
                offset = doc_subwords - (word_indices[0] - 2 - tid)
                results.append([]), b_previous.append([]), b_mentions.append([]), b_refs.append([])
                for doc_mention in doc_mentions:
                    if doc_mention[0] < offset: continue
                    b_previous[-1].append([doc_mention[0] - offset + 1 + tid, doc_mention[1] - offset + 1 + tid])
                    b_refs[-1].append(doc_mention[2])
                for mention in mentions:
                    results[-1].append(mention)
                    b_refs[-1].append(mention)
                    b_mentions[-1].append([word_indices[mention[0]], word_indices[mention[1]]])
                    doc_mentions.append([doc_subwords + word_indices[mention[0]] - word_indices[0],
                                         doc_subwords + word_indices[mention[1]] - word_indices[0], mention])
                doc_subwords += word_indices[-1] - word_indices[0]

            # Decode antecedents
            if sum(len(mentions) for mentions in b_mentions) == 0: continue
            b_antecedents = self.compute_antecedents(
                b_embeddings, tf.ragged.constant(b_previous, dtype=tf.int32, ragged_rank=1),
                tf.ragged.constant(b_mentions, dtype=tf.int32, ragged_rank=1))
            for b in range(b_size):
                len_prev, mentions, refs, antecedents = len(b_previous[b]), b_mentions[b], b_refs[b], b_antecedents[b].numpy()
                for i in range(len(mentions)):
                    j = i - 1
                    while j >= 0 and mentions[j][0] == mentions[i][0]:
                        antecedents[i, j + len_prev] = antecedents[i, i + len_prev] - 1
                        j -= 1
                    j = np.argmax(antecedents[i, :i + len_prev + 1])
                    if j == i + len_prev:
                        entities += 1
                        refs[i + len_prev][2] = entities
                    else:
                        refs[i + len_prev][2] = refs[j][2]

        return results

    def callback(self, epoch: int, datasets: list[tuple[Dataset, tf.data.Dataset]], evaluate: bool) -> None:
        for dataset, pipeline in datasets:
            mentions = self.predict(dataset, pipeline)
            path = os.path.join(self._args.logdir, f"{os.path.splitext(os.path.basename(dataset._path))[0]}.{epoch:02d}.conllu")
            headsonly_path = f"{path.removesuffix('.conllu')}.headsonly.conllu"
            dataset.save_mentions(path, headsonly_path, mentions)
            if evaluate:
                for eval_path in [path, headsonly_path]:
                    os.system(f"./corefud-score.sh '{dataset._path}' '{eval_path}'")


def main(args: argparse.Namespace) -> None:
    # Fix random seeds and threads
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    if args.debug:
        tf.config.run_functions_eagerly(True)
        tf.data.experimental.enable_debug_mode()

    # Create logdir name and dump options
    args.logdir = os.path.join("logs", "{}{}-{}-{}-{}".format(
        args.exp + (args.exp and "-"),
        os.path.splitext(os.path.basename(globals().get("__file__", "notebook")))[0],
        os.environ.get("JOB_ID", ""),
        datetime.datetime.now().strftime("%y%m%d_%H%M%S"),
        ",".join(("{}={}".format(
            re.sub("(.)[^_]*_?", r"\1", k),
            ",".join(re.sub(r"^.*/", "", str(x)) for x in ((v if len(v) <= 1 else [v[0], "..."]) if isinstance(v, list) else [v])),
        ) for k, v in sorted(vars(args).items())))
    ))
    os.makedirs(args.logdir)
    shutil.copy2(__file__, os.path.join(args.logdir, os.path.basename(__file__)))
    with open(os.path.join(args.logdir, "options.json"), "w") as json_file:
        json.dump(vars(args), json_file, sort_keys=True, ensure_ascii=False, indent=2)
    print(json.dumps(vars(args), sort_keys=True, ensure_ascii=False, indent=2))

    # Load the data
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.bert)

    trains = [Dataset(f"{path}-corefud-train.conllu", tokenizer, args.treebank_id * i) for i, path in enumerate(args.data + args.train, 1)]
    devs = [Dataset(f"{path}-corefud-dev.conllu", tokenizer, args.treebank_id * i) for i, path in enumerate(args.data, 1)]
    tests = [Dataset(f"{path}-corefud-test.conllu", tokenizer, args.treebank_id * i) for i, path in enumerate(args.data, 1)]

    tags = Dataset.create_tags(trains)
    with open(os.path.join(args.logdir, "tags.txt"), "w") as tags_file:
        for tag in tags:
            print(tag, file=tags_file)
    tags_map = {tag: i for i, tag in enumerate(tags)}

    strategy_scope = tf.distribute.MirroredStrategy().scope() if len(tf.config.list_physical_devices("GPU")) > 1 else contextlib.nullcontext()
    with strategy_scope:
        # Create pipelines
        def batch(pipeline, drop_remainder=False):
            return pipeline.apply(tf.data.experimental.dense_to_ragged_batch(args.batch_size, drop_remainder)).prefetch(tf.data.AUTOTUNE)
        trains = [train.pipeline(tags_map, True, args) for train in trains]
        if args.resample:
            steps, *ratios = args.resample
            assert len(ratios) == len(trains)
            ratios = [ratio / sum(ratios) for ratio in ratios]
            trains = [train.shuffle(len(train)).repeat().take(1 + int(steps * args.batch_size * ratio))
                      for train, ratio in zip(trains, ratios)]
        train = functools.reduce(lambda x, y: x.concatenate(y), trains)
        train = batch(train.shuffle(len(train), seed=args.seed), drop_remainder=True)
        devs = [(dev, batch(dev.pipeline(tags_map, False, args))) for dev in devs]
        tests = [(test, batch(test.pipeline(tags_map, False, args))) for test in tests]

        # Create and train the model
        model = Model(tags, args)
        model.compile(train)
        model.fit(train, epochs=args.epochs, verbose=os.environ.get("VERBOSE", 2), callbacks=[
            tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, _: model.callback(epoch, devs, evaluate=True)),
            tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, _: model.callback(epoch, tests, evaluate=False)),
        ])


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
