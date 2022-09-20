import torch
import torch.nn as nn 
import numpy as np

class LongShortTermMemory(nn.Module):
    def __init__(self, encoding_size, emoji_size):
        super(LongShortTermMemory, self).__init__()

        self.lstm = nn.LSTM(encoding_size, 128) # 128 state size
        self.dense = nn.Linear(128, emoji_size) # 128 state size
        
    def reset(self):
        zero_state = torch.zeros(1,1,128)
        self.hidden_state = zero_state
        self.cell_state = zero_state
        
    def logits(self, x):
        out, (self.hidden_state, self.cell_state) = self.lstm(x, (self.hidden_state, self.cell_state))
        return self.dense(out.reshape(-1, 128))
    
    def f(self, x):
        return torch.softmax(self.logits(x), dim = 1)
    
    def loss(self, x, y):
        return nn.functional.cross_entropy(self.logits(x), y.argmax(1))



index_to_char = [' ', 'h', 'a', 't', 'r', 'c', 'f', 'l', 'm', 'p', 's', 'o', 'n']

char_encodings = np.eye(len(index_to_char))
encoding_size = len(char_encodings)

emojis = {
    'hat': '\U0001F3A9',
    'rat': '\U0001F408',
    'cat': '\U0001F400',
    'flat': '\U0001F3E2',
    'matt': '\U0001F468',
    'cap': '\U0001F9E2',
    'son': '\U0001F466'
}

index_emoji = [emojis['hat'],emojis['cat'],emojis['rat'],emojis['flat'],emojis['matt'],emojis['cap'],emojis['son']]
emoji_encodings = np.eye(len(index_emoji)) # np.eye-> unit matrix
emoji_size = len(index_emoji)

x_train = torch.tensor([[[char_encodings[1]], [char_encodings[2]], [char_encodings[3]], [char_encodings[0]]],  # "hat "
                        [[char_encodings[4]], [char_encodings[2]], [char_encodings[3]], [char_encodings[0]]],  # "rat "
                        [[char_encodings[5]], [char_encodings[2]], [char_encodings[3]], [char_encodings[0]]],  # "cat "
                        [[char_encodings[6]], [char_encodings[7]], [char_encodings[2]], [char_encodings[3]]],  # "flat"
                        [[char_encodings[8]], [char_encodings[2]], [char_encodings[3]], [char_encodings[3]]],  # "matt"
                        [[char_encodings[5]], [char_encodings[2]], [char_encodings[9]], [char_encodings[0]]],  # "cap "
                        [[char_encodings[10]], [char_encodings[11]], [char_encodings[12]], [char_encodings[0]]]],  # "son "
                       dtype=torch.float)
y_train = torch.tensor([[emoji_encodings[0], emoji_encodings[0], emoji_encodings[0], emoji_encodings[0]],  # "hat "
                        [emoji_encodings[1], emoji_encodings[1], emoji_encodings[1], emoji_encodings[1]],  # "rat "
                        [emoji_encodings[2], emoji_encodings[2], emoji_encodings[2], emoji_encodings[2]],  # "cat "
                        [emoji_encodings[3], emoji_encodings[3], emoji_encodings[3], emoji_encodings[3]],  # "flat"
                        [emoji_encodings[4], emoji_encodings[4], emoji_encodings[4], emoji_encodings[4]],  # "matt"
                        [emoji_encodings[5], emoji_encodings[5], emoji_encodings[5], emoji_encodings[5]],  # "cap "
                        [emoji_encodings[6], emoji_encodings[6], emoji_encodings[6], emoji_encodings[6]]],  # "son "
                       dtype=torch.float)

model = LongShortTermMemory(encoding_size , emoji_size)

def generate(s):
    model.reset()
    for i in range(len(s)):
        index_char = index_to_char.index(s[i])
        y = model.f(torch.tensor([[char_encodings[index_char]]], dtype=torch.float))
        if i == len(s)-1:
            print(index_emoji[y.argmax(1)], ": ", s)
            
            
optimizer = torch.optim.RMSprop(model.parameters(), 0.0001)

for epoch in range (500):
    for i in range(x_train.size()[0]):
        model.reset()
        loss = model.loss(x_train[i], y_train[i])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        
generate("rat")
generate("cat")
generate("hat")
generate("flat")
generate("matt")
generate("cap")
generate("son")
generate("rt")
generate("at")
