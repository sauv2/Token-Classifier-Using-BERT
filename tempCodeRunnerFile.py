        mask = (targets != tokenizer.pad_token_id).bool()

        outputs_masked = outputs.logits.view(-1)
        targets_masked = targets.view(-1)

        valid_indices = mask.nonzero(as_tuple=True)

        if valid_indices[0].numel() > 0:
            outputs_masked = outputs_masked[valid_indices]
            targets_masked = targets_masked[valid_indices]
            loss = loss_fn(outputs_masked, targets_masked)
        else:
            loss = torch.tensor(0.0, requires_grad=True)