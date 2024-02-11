import torch
import torchvision.datasets as DS
import torchvision.transforms as transform


# Sample test
# load MNIST dataset
mnist = DS.MNIST('./data',train=True,transform=transform.ToTensor(),download=True)

def myBatcher(batch_size):
    batch_list = []
    num_batches = int(len(mnist)/batch_size)
    
    for j in range(num_batches):
        batch_x = torch.zeros(batch_size,784)
        batch_v = torch.zeros(batch_size)
        for i in range(batch_size):
            batch_x[i] = mnist[j*batch_size + i][0].reshape(784)
            batch_v[i] = mnist[j*batch_size + i][1]
            
        batch = (batch_x,batch_v)
        batch_list.append(batch)
    return batch_list


def list_check(list_of_items):
    for index, item in enumerate(list_of_items):
        print(f"Item {index}: Type = {type(item)}", end='')
        # If the item is a list or a tuple, print its length and the type of its elements
        if isinstance(item, (list, tuple)):
            print(f", Length = {len(item)}", end='')
            # Optionally, dive deeper into the elements of the list/tuple
            for i, sub_item in enumerate(item):
                print(f"\n    Sub-item {i}: Type = {type(sub_item)}", end='')
                # If it's a tensor, print its shape
                if hasattr(sub_item, 'shape'):
                    print(f", Shape = {sub_item.shape}", end='')
                # If it's another list or tuple, you could recursively dive deeper
                # but here we'll just print the length
                elif isinstance(sub_item, (list, tuple)):
                    print(f", Length = {len(sub_item)}", end='')
                # Add more type checks as needed (e.g., for dictionaries)
        # If the item is a tensor, directly print its shape
        elif hasattr(item, 'shape'):
            print(f", Shape = {item.shape}", end='')
        # Add more type checks if you're expecting other types
        print()  # Newline for clarity