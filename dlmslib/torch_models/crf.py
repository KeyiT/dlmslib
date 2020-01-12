import torch
from torch import nn


class ChainCRF(nn.Module):

    """
    This class implements linear-chained Conditional Random Field model.
    """

    def __init__(self, input_size, num_labels, bigram):
        """
        Args:
            input_size: int
                the dimension of the input.
            num_labels: int
                the number of labels of the crf layer
            bigram: bool
                if apply bi-gram parameter.
        """

        super(ChainCRF, self).__init__()
        self.input_size = input_size
        self.num_labels = num_labels + 1
        self.pad_label_id = num_labels
        self.bigram = bigram

        # state weight tensor
        self.state_nn = nn.Linear(input_size, self.num_labels)
        if bigram:
            # transition weight tensor
            self.trans_nn = nn.Linear(input_size, self.num_labels * self.num_labels)
            self.register_parameter('trans_matrix', None)
        else:
            self.trans_nn = None
            self.trans_matrix = nn.Parameter(torch.Tensor(self.num_labels, self.num_labels))

        self.reset_parameters()

    def forward(self, input_, mask=None, leading_step_to_ignore=0):
        """
                This function recieves the input and decodes the best sequences of labels by the Viterbi algorithm.
                Args:
                    input_: Tensor
                        the input tensor with shape = [batch, length, input_size]
                    mask: Tensor or None
                        the mask tensor with shape = [batch, length]
                    leading_step_to_ignore: int
                        number of symbolic labels leading in type alphabets (set it to 0 if you are not sure)

                Returns: Returns: Tensor, Tensor
                    The first element is the decoding results in shape [batch, length].
                    The second element is the corresponding probability of the decoding results in shape [batch].

                """
        return self.viterbi_decode(input_, mask=mask, kbest=1, leading_step_to_ignore=leading_step_to_ignore)

    def reset_parameters(self):
        nn.init.constant_(self.state_nn.bias, 0.)
        if self.bigram:
            nn.init.xavier_uniform_(self.trans_nn.weight)
            nn.init.constant_(self.trans_nn.bias, 0.)
        else:
            nn.init.normal_(self.trans_matrix)

    def neg_log_likelihood(self, input_, target, mask=None):
        """
                This function calculates the CRF negative log likelihood loss.
                Args:
                    input_: Tensor
                        the input tensor with shape = [batch, length, input_size]
                    target: Tensor
                        the tensor of target labels with shape [batch, length]
                    mask:Tensor or None
                        the mask tensor with shape = [batch, length]

                Returns: Tensor
                        A 1D tensor for negative log likelihood loss with shape = [batch].
                """
        stepwise_factor_logs = self.stepwise_potential(input_, mask=mask)

        # shape = [length, batch, num_label, num_label]
        stepwise_factor_logs_transpose = stepwise_factor_logs.transpose(0, 1)

        return self.partition_function(stepwise_factor_logs_transpose, mask) - \
               self.target_potential(stepwise_factor_logs_transpose, target)

    def viterbi_decode(self, input_, mask=None, kbest=1, leading_step_to_ignore=0):
        """
        This function decodes the k best sequences of labels by the Viterbi algorithm.
        Args:
            input_: Tensor
                the input tensor with shape = [batch, length, input_size]
            mask: Tensor or None
                the mask tensor with shape = [batch, length]
            leading_step_to_ignore: int
                number of symbolic labels leading in type alphabets (set it to 0 if you are not sure)
            kbest: int
                number of hypotheses to return

        Returns: Tensor, Tensor
            The first element is the decoding results in shape [batch, length, kbest].
            The second element is the corresponding probability of the top-k decoding results in shape [batch, kbest].

        """

        stepwise_factor_logs = self.stepwise_potential(input_, mask=mask)

        # Input should be provided as (n_batch, n_time_steps, num_labels, num_labels)
        # For convenience, we need to dim shuffle to (n_time_steps, n_batch, num_labels, num_labels)
        stepwise_factor_logs_transpose = stepwise_factor_logs.transpose(0, 1)

        partition_function = self.partition_function(stepwise_factor_logs_transpose, mask)

        # the last row and column is the tag for pad symbol. reduce these two dimensions by 1 to remove that.
        # also remove the first #symbolic rows and columns.
        # now the shape of energies_shuffled is [n_time_steps, b_batch, t, t] where t = num_labels - #symbolic - 1.
        stepwise_factor_logs_transpose = stepwise_factor_logs_transpose[:, :, leading_step_to_ignore:-1, leading_step_to_ignore:-1]

        length, batch_size, num_labels, _ = stepwise_factor_logs_transpose.size()

        batch_index = torch.arange(0, batch_size,
                                   device=stepwise_factor_logs_transpose.device,
                                   dtype=torch.long)

        if kbest <= 1:
            back_pointer = batch_index.new_zeros(length, batch_size)
            pointer = batch_index.new_zeros(length, batch_size, num_labels)
            pi = stepwise_factor_logs_transpose.new_zeros([length, batch_size, num_labels])
            pi[0] = stepwise_factor_logs[:, 0, -1, leading_step_to_ignore:-1]
            pointer[0] = -1
            for t in range(1, length):
                pi_prev = pi[t - 1]
                pi[t], pointer[t] = torch.max(stepwise_factor_logs_transpose[t] + pi_prev.unsqueeze(2), dim=1)

            score, back_pointer[-1] = torch.max(pi[-1], dim=1)
            for t in reversed(range(length - 1)):
                pointer_last = pointer[t + 1]
                back_pointer[t] = pointer_last[batch_index, back_pointer[t + 1]]

            back_pointer = back_pointer.transpose(0, 1).unsqueeze(2) + leading_step_to_ignore
            score = score.unsqueeze(1)
        else:
            path_indices, path_score = self.max_sum_forward_messages(
                stepwise_factor_logs_transpose, mask=mask, kbest=kbest, leading_step_to_ignore=leading_step_to_ignore
            )

            path_score = path_score.view(batch_size, -1)
            # assert path_score.size() == (batch_size, max_k * num_labels)
            max_k = min(path_score.size()[1], kbest)
            back_pointer = batch_index.new_zeros(length, batch_size, max_k)
            score, back_pointer[-1] = torch.topk(path_score, k=max_k, dim=1)
            # assert scores.size() == (batch_size, max_k)

            for t in reversed(range(length - 1)):
                back_pointer[t] = path_indices[t].view(batch_size, -1)\
                    .gather(dim=1, index=back_pointer[t+1])\
                    .squeeze()

            back_pointer = back_pointer % num_labels
            back_pointer = back_pointer.transpose(0, 1) + leading_step_to_ignore

        return back_pointer, torch.clamp(torch.exp(score - partition_function.unsqueeze(1)), 0, 1)

    def stepwise_potential(self, input_, mask=None):
        """
                This function receives the input and outputs the summation of the log-scale emit factors
                log[Psi(X_t, Y_t)] and the transition factors log[Psi(Y_t, Y_{t+1})] at each step.
                Args:
                    input_: Tensor
                        the input tensor with shape = [batch, length, input_size]
                    mask: Tensor or None
                        the mask tensor with shape = [batch, length]

                Returns: Tensor
                    the energy tensor with shape = [batch, length, num_label, num_label]

        """
        batch, length, _ = input_.size()

        # compute out_s by tensor dot [batch, length, input_size] * [input_size, num_label]
        # thus out_s should be [batch, length, num_label] --> [batch, length, num_label, 1]
        out_s = self.state_nn(input_).unsqueeze(2)

        if self.bigram:
            # compute out_s by tensor dot: [batch, length, input_size] * [input_size, num_label * num_label]
            # the output should be [batch, length, num_label,  num_label]
            out_t = self.trans_nn(input_).view(batch, length, self.num_labels, self.num_labels)
            output = out_t + out_s
        else:
            # [batch, length, num_label, num_label]
            output = self.trans_matrix + out_s

        output = output.float()
        if mask is not None:
            output = output * mask.unsqueeze(2).unsqueeze(3)

        return output

    @staticmethod
    def partition_function(stepwise_factor_logs_transpose, mask=None):
        """
                This function calculates the log-scale CRF partition function by the prod-sum-forward-pass algorithm.
                Args:
                    stepwise_factor_logs_transpose: Tensor
                        The stepwise log-scale factor tensor with shape = [length, batch, num_label, num_label]
                    mask: Tensor or None
                        The mask tensor with shape = [batch, length]

                Returns: Tensor
                        A 1D tensor for the log-scale CRF partition function with shape = [batch].
                """
        # shape = [length, batch, num_label, num_label]
        mask_transpose = None
        if mask is not None:
            mask_transpose = mask.unsqueeze(2).transpose(0, 1)

        # shape = [batch, num_label]
        lambda_t = None
        length = stepwise_factor_logs_transpose.size()[0]

        for t in range(length):
            if t == 0:
                lambda_t = stepwise_factor_logs_transpose[t, :, -1, :]
            else:
                # shape = [batch, num_label]
                lambda_tplus1 = log_sum_exp(stepwise_factor_logs_transpose[t] + lambda_t.unsqueeze(2), dim=1)
                if mask_transpose is None:
                    lambda_t = lambda_tplus1
                else:
                    mask_t = mask_transpose[t]
                    lambda_t = lambda_t + (lambda_tplus1 - lambda_t) * mask_t

        return log_sum_exp(lambda_t, dim=1)

    @staticmethod
    def target_potential(stepwise_factor_logs_transpose, target):
        """
        This function calculates the log-scale un-normalized likelihood of the hypothesis log[p(Y|X;theta)].
        Args:
            stepwise_factor_logs_transpose: Tensor
                the energy tensor with shape = [length, batch, num_label, num_label]
            target: Tensor
                the tensor of target labels with shape [batch, length]

        Returns: Tensor
                A 1D tensor for the log-scale un-normalized likelihood of the hypothesis p(Y|X;theta)
                with shape = [batch].
        """
        length, batch, num_label, _ = stepwise_factor_logs_transpose.size()
        # shape = [length, batch]
        target_transpose = target.transpose(0, 1)

        # shape = [batch]
        batch_index = torch.arange(0, batch).type_as(stepwise_factor_logs_transpose).long()
        prev_label = stepwise_factor_logs_transpose.new_full((batch,), num_label - 1).long()
        likelihood = stepwise_factor_logs_transpose.new_zeros(batch)

        for t in range(length):
            likelihood += stepwise_factor_logs_transpose[t, batch_index, prev_label, target_transpose[t]]
            prev_label = target_transpose[t]

        return likelihood

    @staticmethod
    def max_sum_forward_messages(stepwise_factor_logs_transpose, mask=None, kbest=1, leading_step_to_ignore=0):
        """
        This function calculates the top-k decoding paths and their corresponding scores by the
        max-sum-forward-pass algorithm.

        :param stepwise_factor_logs_transpose: Tensor
                the energy tensor with shape = [length, batch, num_label, num_label]
        :param mask: Tensor or None
                the mask tensor with shape = [batch, length]
        :param kbest: int
                number of hypotheses to return.
        :param leading_step_to_ignore: int
                number of symbolic labels leading in type alphabets (set it to 0 if you are not sure)
        :return: list(Tensor), Tensor
            The first element is the top-k path indices list, the indices tensors are in shape
            [batch_size, max_k, num_labels], the list is in shape [length].
            The second element is the log-scale inference score for the top-k path with the shape =
            [batch_size, max_k, num_labels].

        """
        length, batch_size, num_labels, _ = stepwise_factor_logs_transpose.size()

        path_indices = []
        # shape = (batch_size, max_k, num_labels)
        path_score = stepwise_factor_logs_transpose[0, :, -1, leading_step_to_ignore:-1].unsqueeze(1)

        # shape = (num_labels, num_labels), p(i | j) = 1 / num_labels, p(i | i) = 0
        identity_transition = 1 - torch.eye(num_labels, dtype=mask.dtype, layout=mask.layout, device=mask.device)
        # shape = (n_time_steps, n_batch, num_labels, num_labels)
        identity_transition = identity_transition.unsqueeze(0).unsqueeze(0).expand_as(stepwise_factor_logs_transpose)
        # make masked position as -inf, make other as 0, as we operate in log space
        # log(0) -> -inf, log(1) = 0
        # shape = (n_time_steps, n_batch, num_labels, num_labels)
        identity_masked = identity_transition * (1 - mask.transpose(0, 1)).unsqueeze(2).unsqueeze(3)
        identity_masked[identity_masked == 1] = float('-inf')
        # apply the mask
        stepwise_factor_logs_transpose = stepwise_factor_logs_transpose + identity_masked

        for t in range(1, length):
            # ----- apply lambda_tplus1 = log[Psi_tplus1] + lambda_t -----
            # shape = (batch_size, max_k, num_labels, 1)
            path_score_prev = path_score.unsqueeze(3)
            # shape = (batch_size, max_k, num_labels, num_labels) =
            # (batch_size, max_k, num_labels, 1) + (batch_size, 1, num_labels, 1)
            potentials = path_score_prev + stepwise_factor_logs_transpose[t].unsqueeze(1)
            # shape = (batch_size, max_k * num_labels, num_labels)
            potentials = potentials.view(batch_size, -1, num_labels)

            # ----- find top-k assignment of y_t for lambda_tplus1 -----
            max_k = min(potentials.size()[1], kbest)
            # shape = (batch_size, max_k, num_labels), (batch_size, max_k, num_labels)
            path_score, path_index = torch.topk(potentials, k=max_k, dim=1)
            # assert path_score.size() == (batch_size, max_k, num_labels)
            path_indices.append(path_index)

        return path_indices, path_score


def log_sum_exp(x, dim=None):
    """

    Args:
        x: A pytorch tensor (any dimension will do)
        dim: int or None, over which to perform the summation. `None`, the
             default, performs over all axes.

    Returns: The result of the log(sum(exp(...))) operation.

    """
    if dim is None:
        xmax = x.max()
        xmax_ = x.max()
        return xmax_ + torch.log(torch.exp(x - xmax).sum())
    else:
        xmax, _ = x.max(dim, keepdim=True)
        xmax_, _ = x.max(dim)
        return xmax_ + torch.log(torch.exp(x - xmax).sum(dim))