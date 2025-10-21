from collections import defaultdict

import numpy as np

from beras.core import Diffable, Tensor

class GradientTape:

    def __init__(self):
        # Dictionary mapping the object id of an output Tensor to the Diffable layer it was produced from.
        self.previous_layers: defaultdict[int, Diffable | None] = defaultdict(lambda: None)

    def __enter__(self):
        # When tape scope is entered, all Diffables will point to this tape.
        if Diffable.gradient_tape is not None:
            raise RuntimeError("Cannot nest gradient tape scopes.")

        Diffable.gradient_tape = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # When tape scope is exited, all Diffables will no longer point to this tape.
        Diffable.gradient_tape = None

    def gradient(self, target: Tensor, sources: list[Tensor]) -> list[Tensor]:
        """
        Computes the gradient of the target tensor with respect to the sources.

        :param target: the tensor to compute the gradient of, typically loss output
        :param sources: the list of tensors to compute the gradient with respect to
        In order to use tensors as keys to the dictionary, use the python built-in ID function here: https://docs.python.org/3/library/functions.html#id.
        To find what methods are available on certain objects, reference the cheat sheet
        """

        ### TODO: Populate the grads dictionary with {weight_id, weight_gradient} pairs.

        queue = [target]                    ## Live queue; will be used to propagate backwards via breadth-first-search.
        grads = defaultdict(lambda: None)   ## Grads to be recorded. Initialize to None. Note: stores {id: list[gradients]}
        # Use id(tensor) to get the object id of a tensor object.
        # in the end, your grads dictionary should have the following structure:
        # {id(tensor): [gradient]}

        # What tensor and what gradient is for you to implement!
        # compose_input_gradients and compose_weight_gradients are methods that will be helpful
        grads[id(target)] = np.ones_like(np.asarray(target))

        visited = set()

        while queue:
            out_tensor = queue.pop(0)
            if id(out_tensor) in visited:
                continue
            visited.add(id(out_tensor))

            layer = self.previous_layers[id(out_tensor)]
            if layer is None:
                continue

            upstream_grads = grads[id(out_tensor)]
            if upstream_grads is None:
                continue

            upstream_grads = np.asarray(upstream_grads)

            if layer.inputs:
                expected_batch = layer.inputs[0].shape[0]
            elif layer.outputs:
                expected_batch = layer.outputs[0].shape[0]
            else:
                expected_batch = upstream_grads.shape[0] if upstream_grads.ndim > 0 else 1

            if upstream_grads.ndim == 0:
                upstream_grads = np.ones((expected_batch,), dtype=upstream_grads.dtype)
            elif upstream_grads.shape[0] != expected_batch:
                upstream_grads = np.broadcast_to(upstream_grads, (expected_batch, *upstream_grads.shape[1:]))

            weight_grads = layer.compose_weight_gradients([upstream_grads])
            for w, wg in zip(layer.weights, weight_grads):
                if grads[id(w)] is None:
                    grads[id(w)] = np.array(wg, copy=True)
                else:
                    grads[id(w)] += wg

            input_grads = layer.compose_input_gradients([upstream_grads])
            for inp, ig in zip(layer.inputs, input_grads):
                if grads[id(inp)] is None:
                    grads[id(inp)] = np.array(ig, copy=True)
                else:
                    grads[id(inp)] += ig
                queue.append(inp)

        final_grads = []
        for src in sources:
            grad = grads.get(id(src))
            if grad is None:
                grad = np.zeros_like(src)
            final_grads.append(Tensor(grad))

        return final_grads


