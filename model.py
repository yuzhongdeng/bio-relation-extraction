import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup


ENTITY_SEP_TOKEN = '[LINK]'

class BERTCustomModel(object):
        def __init__(self, epochs=5, batch_size=64, device=None):
            self.tokenizer = tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            special_tokens_dict = {'additional_special_tokens': [ENTITY_SEP_TOKEN,]}
            self.tokenizer.add_special_tokens(special_tokens_dict)

            self.model = BertForSequenceClassification.from_pretrained(
                            "bert-base-uncased",
                            num_labels = 2,
                            output_attentions = False,
                            output_hidden_states = False
                        )
            self.model.resize_token_embeddings(len(self.tokenizer))

            self.optimizer = AdamW(self.model.parameters(), lr = 1e-6)

            self.epochs = epochs
            self.batch_size = batch_size
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = torch.device(device)
            print("[Model] Current device:", self.device)

        def tokenize(self, X):
            '''
            Inputs: - X : list of tuples (entity_pair, context)
            '''
            # Tokenize all of the sentences and map the tokens to thier word IDs.
            input_ids = []

            # For every sentence...
            for entity_pair, context in X:
                encoded_dict = self.tokenizer.encode_plus(entity_pair, context,
                                                      add_special_tokens = True,
                                                      padding='longest',
                                                      return_tensors = 'pt')
                
                # Add the encoded sentence to the list.
                input_ids.append(encoded_dict['input_ids'])
                
            # Convert the lists into tensors.
            input_ids = torch.cat(input_ids, dim=0)
            return input_ids

        def fit(self, X, y):
            """
            Inputs: - X : list of tuples (entity_pair, context)
                    - y : binary, 0 or 1
            Returns: Self
            """
            # ******* Tokenize *******
            input_ids = self.tokenize(X)
            labels = torch.tensor(y)

            # ******* Datasets *******
            dataset = TensorDataset(input_ids, labels)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            # ******* Training *******
            total_steps = len(dataloader) * self.epochs
            scheduler = get_linear_schedule_with_warmup(self.optimizer, 
                                                        num_warmup_steps = 0,
                                                        num_training_steps = total_steps)

            self.model.to(device=self.device)  # move the model parameters to CPU/GPU
            self.model.train()  # put model to training mode
            # Measure how long the training epoch takes.
            t0 = time.time()
            for e in range(self.epochs):
                epoch_error = 0.0
                print("")
                print('======== Epoch {:} / {:} ========'.format(e + 1, self.epochs))
                print('Training...')
                for t, batch in enumerate(dataloader):
                    b_input_ids = batch[0].to(device=self.device)
                    b_labels = batch[1].to(device=self.device)

                    self.model.zero_grad()

                    loss, logits = self.model(b_input_ids, labels=b_labels)
                    epoch_error += loss.item()

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                    # Update parameters and take a step using the computed gradient.
                    self.optimizer.step()

                    # Update the learning rate.
                    scheduler.step()

                    # Progress update every 100 batches.
                    if (t+1) % 100 == 0:
                      # Calculate elapsed time in minutes.
                      elapsed = (time.time() - t0) / 60
                      print('   Batch {} / {}. Elapsed: {} mins.'.format(
                        t+1, len(dataloader), elapsed))
                      
                # training loss and time logging.
                epoch_error = epoch_error / len(dataloader)
                training_time = (time.time() - t0) / 60
                print("Average training loss: {0:.2f}".format(epoch_error))
                print("Training epcoh took: {0:.2f} mins.".format(training_time))

            return self


        def predict(self, X):
            """Predicted labels for the examples in `X`. These are converted
            from the integers that PyTorch needs back to their original
            values in `self.classes_`.

            Input: X : np.array

            Returns: list of length len(X)

            """
            input_ids = self.tokenize(X)
            self.model.eval()
            
            print("")
            print('======== Predictions ========')

            with torch.no_grad():
                self.model.to(self.device)
                input_ids = input_ids.to(self.device)
                logits = self.model(input_ids)[0]
            
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            predictions = [i for i in probs.argmax(axis=1)]
            print(" Finished. Input length: {}, Output length: {}".format(len(X), len(predictions)))
            return predictions