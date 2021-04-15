import syft as sy


class AggregationServer:

    def __init__(self, model):
        self.duet = sy.launch_duet(loopback=True)
        self.model = model

    def train(self, torch_ref, train_loader, optimizer, epoch, args, train_data_length):
        # + 0.5 lets us math.ceil without the import
        train_batches = round((train_data_length / args["batch_size"]) + 0.5)
        print(f"> Running train in {train_batches} batches")
        if self.model.is_local:
            print("Training requires remote model")
            return

        self.model.train()

        for batch_idx, data in enumerate(train_loader):
            data_ptr, target_ptr = data[0], data[1]
            optimizer.zero_grad()
            output = self.model(data_ptr)
            loss = torch_ref.nn.functional.nll_loss(output, target_ptr)
            loss.backward()
            optimizer.step()
            loss_item = loss.item()
            train_loss = loss_item.resolve_pointer_type()
            if batch_idx % args["log_interval"] == 0:
                local_loss = None
                local_loss = train_loss.get(
                    reason="To evaluate training progress",
                    request_block=True,
                    timeout_secs=5
                )
                if local_loss is not None:
                    print("Train Epoch: {} {} {:.4}".format(epoch, batch_idx, local_loss))
                else:
                    print("Train Epoch: {} {} ?".format(epoch, batch_idx))
            if batch_idx >= train_batches - 1:
                print("batch_idx >= train_batches, breaking")
                break
            if args["dry_run"]:
                break
