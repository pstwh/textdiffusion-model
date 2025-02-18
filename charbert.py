import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from collections import Counter
from itertools import chain
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by the number of heads"
        
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.fc_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        
        Q = self.query(x)
        K = self.key(x)  
        V = self.value(x)

        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)

        return self.fc_out(attn_output)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, x):
        attn_out = self.attention(x)
        x = self.norm1(x + attn_out)

        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)

        return x

class CharBERT(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, num_heads=4, num_layers=2, max_len=64, num_classes=2):
        super(CharBERT, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_embedding = nn.Embedding(max_len, embed_dim)
        
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, hidden_dim) for _ in range(num_layers)
        ])
        
        self.output_proj = nn.Linear(embed_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        batch_size, seq_len = x.shape
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        x = self.embedding(x) + self.positional_embedding(positions)

        for layer in self.transformer_layers:
            x = layer(x)

        x = x.mean(dim=1)
        x = self.output_proj(x)
        x = self.classifier(x)
        return x
    

if __name__ == '__main__':
    df = pd.read_csv('data/train2.csv')
    df['text'] = df['sms']
    del df['sms']
    
    df = df[['text', 'label']]
    df = df.map(str)

    class TextDataset(Dataset):
        def __init__(self, dataframe, vocab=None, max_len=64):
            self.data = dataframe
            self.max_len = max_len
            
            if vocab is None:
                self.vocab = self.build_vocab(self.data['text'])
            else:
                self.vocab = vocab
            
            self.vocab_size = len(self.vocab)
            self.label_map = {label: idx for idx, label in enumerate(sorted(self.data['label'].unique()))}
            
        def build_vocab(self, texts):
            all_chars = list(chain.from_iterable(texts))
            counter = Counter(all_chars)
            vocab = {char: idx + 1 for idx, (char, _) in enumerate(counter.most_common())}
            vocab['<PAD>'] = 0
            return vocab
        
        def encode_text(self, text):
            encoded = [self.vocab.get(char, 1) for char in text[:self.max_len]]
            if len(encoded) < self.max_len:
                encoded += [0] * (self.max_len - len(encoded))
            return encoded
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            text = self.data.iloc[idx]['text']
            sentiment = self.data.iloc[idx]['label']
            
            encoded_text = torch.tensor(self.encode_text(text), dtype=torch.long)
            label = torch.tensor(self.label_map[sentiment], dtype=torch.long)
            
            return encoded_text, label
        
    def create_dataloaders(df, batch_size=32):
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
        vocab = TextDataset(train_df).vocab
        train_dataset = TextDataset(train_df, vocab=vocab, max_len=128)
        val_dataset = TextDataset(val_df, vocab=vocab, max_len=128)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, vocab

    train_loader, val_loader, vocab = create_dataloaders(df)

    def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, device, epochs=10):
        model.to(device)
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            all_preds = []
            all_labels = []

            for texts, labels in train_loader:
                texts, labels = texts.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(texts)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            train_accuracy = accuracy_score(all_labels, all_preds)

            model.eval()
            val_loss = 0
            val_preds = []
            val_labels = []
            val_probs = []
            
            with torch.no_grad():
                for texts, labels in val_loader:
                    texts, labels = texts.to(device), labels.to(device)
                    outputs = model(texts)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    preds = outputs.argmax(dim=1)
                    probs = F.softmax(outputs, dim=1)[:, 1]
                    
                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())
                    val_probs.extend(probs.cpu().numpy())

            val_accuracy = accuracy_score(val_labels, val_preds)
            num_classes = len(set(val_labels))

            if num_classes == 2:
                val_roc_auc = roc_auc_score(val_labels, val_probs)
                val_f1 = f1_score(val_labels, val_preds)
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {total_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, "
                    f"Val ROC AUC: {val_roc_auc:.4f}, Val F1 Score: {val_f1:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {total_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
                
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CharBERT(vocab_size=len(vocab), embed_dim=64, hidden_dim=128, num_heads=16, num_layers=2, max_len=128, num_classes=len(df['label'].unique())).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    total_params = sum(p.numel() for p in model.parameters())

    print(f"Total parameters: {total_params}")

    train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, device, epochs=50)