
# memory_budget.py

# CIFAR-10 image storage
RAW_IMAGE_BYTES = 32 * 32 * 3
LABEL_BYTES = 1
NUM_CLASSES = 10
LOGIT_BYTES = NUM_CLASSES * 4 
LATENT_SHAPE = (8, 8, 8)
LATENT_BYTES = 8 * 8 * 8 * 4  

# Experimental memory limits
BYTE_BUDGETS = {
    "small": 32 * 1024,
    "medium": 64 * 1024,
    "large": 128 * 1024
}

# Total bytes required per replay sample
RAW_SAMPLE_BYTES = RAW_IMAGE_BYTES + LABEL_BYTES + LOGIT_BYTES

COMPRESSED_SAMPLE_BYTES = (
    LATENT_BYTES + LABEL_BYTES + LOGIT_BYTES
)

print("Raw sample:", RAW_SAMPLE_BYTES, "bytes")
print("Compressed sample:", COMPRESSED_SAMPLE_BYTES, "bytes")

print("\nReplay buffer capacities:")

for name, budget in BYTE_BUDGETS.items():
    raw_capacity = budget // RAW_SAMPLE_BYTES
    compressed_capacity = budget // COMPRESSED_SAMPLE_BYTES

    print(f"\n{name.capitalize()} budget: {budget} bytes")
    print("Raw samples:", raw_capacity)
    print("Compressed samples:", compressed_capacity)
