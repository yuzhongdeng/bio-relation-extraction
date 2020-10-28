import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from transformers import (
    RobertaTokenizer, 
    RobertaForSequenceClassification,
    BertTokenizer, 
    BertForSequenceClassification, 
    AdamW, 
    get_linear_schedule_with_warmup
)

from sklearn.metrics import classification_report

ENTITY_SEP_TOKEN = '[ESEP]'
CONTEXT_SEP_TOKEN = '[CSEP]'

class BERTCustomModel(object):
        def __init__(self, epochs=5, batch_size=64, device=None):
            self.tokenizer = tokenizer = RobertaTokenizer.from_pretrained('bert-base-uncased')
            special_tokens_dict = {'additional_special_tokens': [ENTITY_SEP_TOKEN, CONTEXT_SEP_TOKEN]}
            self.tokenizer.add_special_tokens(special_tokens_dict)

            self.model = RobertaForSequenceClassification.from_pretrained(
                            "roberta-base",
                            num_labels = 2,
                            output_attentions = False,
                            output_hidden_states = False
                        )
            self.model.resize_token_embeddings(len(self.tokenizer))
            self.optimizer = AdamW(self.model.parameters(), lr = 1e-4)

            self.epochs = epochs
            self.batch_size = batch_size

            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = torch.device(device)
            print("[Model] Current device:", self.device)

        def tokenize(self, X):
            # Tokenize all of the sentences and map the tokens to thier word IDs.
            input_ids = []
            attention_masks = []

            # For every sentence...
            for entities, context in X:
                encoded_dict = self.tokenizer.encode_plus(entities, context,
                                                      add_special_tokens = True,
                                                      max_length = 128, 
                                                      padding='max_length',
                                                      truncation = True,
                                                      return_attention_mask = True,
                                                      return_tensors = 'pt')
                
                # Add the encoded sentence to the list.
                input_ids.append(encoded_dict['input_ids'])
                # And its attention mask (simply differentiates padding from non-padding).
                attention_masks.append(encoded_dict['attention_mask'])
                
            # Convert the lists into tensors.
            input_ids = torch.cat(input_ids, dim=0)
            attention_masks = torch.cat(attention_masks, dim=0)
            return input_ids, attention_masks

        def train(self, X, y, X_dev, y_dev):
            """
            Inputs: - X : list of str (entity_a, entity_b, context)
                    - y : binary, 0 or 1
            Returns: Self
            """
            # ******* Tokenize *******
            input_ids, attention_masks = self.tokenize(X)
            labels = torch.tensor(y)

            # ******* Datasets *******
            dataset = TensorDataset(input_ids, attention_masks, labels)
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
                print(f'======== Epoch {e + 1} / {self.epochs} ========')
                print('Training...')
                for t, batch in enumerate(dataloader):
                    b_ids = batch[0].to(device=self.device)
                    b_masks = batch[1].to(device=self.device)
                    b_labels = batch[2].to(device=self.device)

                    self.model.zero_grad()

                    loss, logits = self.model(b_ids, attention_mask=b_masks, labels=b_labels)
                    epoch_error += loss.item()

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                    # Update parameters and take a step using the computed gradient.
                    self.optimizer.step()

                    # Update the learning rate.
                    scheduler.step()

                    # Progress update every 100 batches.
                    if (t+1) % 50 == 0:
                      # Calculate elapsed time in minutes.
                      elapsed = (time.time() - t0) / 60
                      print(f'\tBatch {t+1:>3} / {len(dataloader)}. Elapsed: {elapsed:.2f} mins.')
                      
                # training loss and time logging.
                epoch_error = epoch_error / len(dataloader)
                training_time = (time.time() - t0) / 60
                print(f"Average training loss: {epoch_error:.2f}")
                print(f"Training epcoh took: {training_time:.2f} mins.")


                print("")
                print("Running Validation...")
                self.test(X_dev, y_dev)

            return self


        def test(self, X, y):
            """Predicted labels for the examples in `X`. These are converted
            from the integers that PyTorch needs back to their original
            values in `self.classes_`.

            Input: X : np.array

            Returns: list of length len(X)

            """
            input_ids, attention_masks = self.tokenize(X)
            self.model.eval()
            
            with torch.no_grad():
                input_ids = input_ids.to(self.device)
                attention_masks = attention_masks.to(self.device)
                logits = self.model(input_ids, attention_mask=attention_masks)[0]
            
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            predictions = [i for i in probs.argmax(axis=1)]
            
            print(f" Finished. Input length: {len(X)}, Output length: {len(predictions)}")
            print(classification_report(y, predictions, digits=3))
            
            return predictions