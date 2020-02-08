from itertools import zip_longest
import functools
import numpy as np
from samplers import register_multi_sampler


def annotate_data_with_dataset_id(batch, dataset):
    """Annotates the dataset with identifier set by MultiDataset"""
    return list(zip_with_scalar(batch, dataset.name))


def zip_with_scalar(l, o):
    # l - the list; o - the object
    return zip_longest([o], l, fillvalue=o)


def get_next(iterator):
    try:
        return next(iterator)
    except StopIteration:
        iterator.reset()
        return next(iterator)


class MultiSampler(object):
    @property
    def keys(self):
        self.keys = set()
        for sampler in self.samplers:
            self.keys.update(sampler.dataset.header)

    def __init__(self, samplers):
        self.samplers = samplers

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def reset(self):
        for sampler in self.samplers:
            sampler.reset()

    @classmethod
    def build(cls, samplers, cfg):
        return cls(samplers)


@register_multi_sampler('switching_sampler_shortest')
class SwitchingSamplerShortest(MultiSampler):
    def __iter__(self):
        samplers = self.samplers.copy()
        pick = 0
        len_samplers = len(samplers)
        while len_samplers:
            sampler = samplers[pick]
            try:
                sampler_batch = next(sampler)
                pick = (pick + 1) % len_samplers

            except StopIteration:
                break
            batch = annotate_data_with_dataset_id(sampler_batch, sampler.dataset)
            yield batch

        # we need to reset all samplers,
        # otherwise one sampler is already advanced
        self.reset()

    def __len__(self):
        return min([len(sampler) for sampler in self.samplers]) * len(self.samplers)


@register_multi_sampler('switching_sampler_longest')
class SwitchingSamplerLongest(MultiSampler):
    def __iter__(self):
        samplers = self.samplers.copy()
        pick = 0
        len_samplers = len(samplers)
        while len_samplers:
            sampler = samplers[pick]
            try:
                sampler_batch = next(sampler)
                pick = (pick + 1) % len_samplers
                batch = annotate_data_with_dataset_id(sampler_batch, sampler.dataset)
                yield batch

            except StopIteration:
                sampler.reset()
                samplers.remove(sampler)
                len_samplers = len_samplers - 1
                if len_samplers == 0:
                    break
                pick = pick % len_samplers

        # For longest it is fine not to reset all
    def __len__(self):
        return sum([len(sampler) for sampler in self.samplers])


@register_multi_sampler('random_sampler_shortest')
class RandomSamplerShortest(MultiSampler):
    def __iter__(self):
        samplers = self.samplers.copy()
        len_samplers = len(samplers)

        while True:
            pick = np.random.randint(0, len_samplers)
            sampler = samplers[pick]
            try:
                sampler_batch = next(sampler)
            except StopIteration:
                break

            batch = annotate_data_with_dataset_id(sampler_batch, sampler.dataset)
            yield batch
        self.reset()

    def __len__(self):
        return sum([len(sampler) for sampler in self.samplers])


@register_multi_sampler('random_sampler_longest')
class RandomSamplerLongest(MultiSampler):
    def __iter__(self):
        samplers = self.samplers.copy()
        len_samplers = len(samplers)
        while len_samplers:
            pick = np.random.randint(0, len_samplers)
            sampler = samplers[pick]
            try:
                sampler_batch = next(sampler)
                batch = annotate_data_with_dataset_id(sampler_batch, sampler.dataset)
                yield batch
            except StopIteration:
                sampler.reset()
                samplers.remove(sampler)
                len_samplers = len_samplers - 1


    def __len__(self):
        return sum([len(sampler) for sampler in self.samplers])


@register_multi_sampler('random_sampler_longest_keep')
class RandomSamplerLongestKeep(MultiSampler):
    """
    Keep sampling until all samplers have done one iteration.
    """
    def __iter__(self):
        samplers = self.samplers.copy()
        len_samplers = len(samplers)
        finished_samplers = [False] * len(samplers)
        while not all(finished_samplers):
            pick = np.random.randint(0, len_samplers)
            sampler = samplers[pick]
            try:
                sampler_batch = next(sampler)
                batch = annotate_data_with_dataset_id(sampler_batch, sampler.dataset)
                yield batch
            except StopIteration:
                sampler.reset()
                finished_samplers[pick] = True

    def __len__(self):
        """
        As this is random, it cannot really be predicted.
        """
        return max([len(sampler) for sampler in self.samplers]) * len(self.samplers)



class RandomSamplerWeighted(MultiSampler):
    def __init__(self, samplers, sample_fn):
        super().__init__(samplers)
        self.sample_fn = sample_fn
        self.counter = [0] * len(samplers)


    def __iter__(self):
        samplers = self.samplers.copy()
        finished = [False] * len(samplers)
        while True:
            pick = self.sample_fn()
            sampler = samplers[pick]
            try:
                sampler_batch = next(sampler)
                batch = annotate_data_with_dataset_id(sampler_batch, sampler.dataset)
                yield batch
            except StopIteration:
                finished[pick] = True
                self.counter[pick] += 1
                if all(finished):
                    break
                sampler.reset()

        # TODO Should reset all
        # if we reset all, this means the length is determined by the longest dataset
        # and its probabilitya

        # Target steps per dataset per epoch
        self.reset()
        print(self.counter)

    def __len__(self):
        return sum([len(sampler) for sampler in self.samplers])


@register_multi_sampler('random_sampler_length_weighted')
class RandomSamplerLengthWeighted(RandomSamplerWeighted):
    def __init__(self, samplers, weights):
        """Draws from a distribution where each sampler is 
        weighted with its length and an optional factor."""

        probs = self.calc_probs(samplers, weights)
        self.probs = probs
        sample_fn = functools.partial(np.random.choice, len(samplers), p=probs)
        super().__init__(samplers, sample_fn)

    @staticmethod
    def calc_probs(samplers, weights):
        """
        Calc probs depending on the length of the sampler
        """
        scaled_length = []
        for weight, sampler in zip(weights, samplers):
            scaled_length.append(weight * len(sampler))

        probs = np.array(scaled_length) / sum(scaled_length)
        return probs

    @classmethod
    def build(cls, samplers, cfg):
        weights = cfg.get("weights", [1] * len(samplers))
        if weights == "equal":
            length = []
            for sampler in samplers:
                length.append(len(sampler))

            weights = max(length) / np.array(length)
        assert len(weights) == len(samplers)
        return cls(samplers, weights)


class ConcatenatedSampler(MultiSampler):
    def __iter__(self):
        for i in range(len(self)):
            batch = []
            for sampler_id, sampler in enumerate(self.samplers):
                try:
                    sampler_batch = next(sampler)
                except StopIteration:
                    sampler.reset()
                    sampler_batch = next(sampler)
                sampler_batch = annotate_data_with_dataset_id(sampler_batch, sampler.dataset)
                batch.extend(sampler_batch)
            yield batch

    def __len__(self):
        raise NotImplementedError


@register_multi_sampler('concatenated_longest')
class ConcatenatedSamplerLongest(ConcatenatedSampler):
    def __len__(self):
        return max([len(sampler) for sampler in self.samplers])


@register_multi_sampler('concatenated_shortest')
class ConcatenatedSamplerShortest(ConcatenatedSampler):
    def __len__(self):
        return min([len(sampler) for sampler in self.samplers])
