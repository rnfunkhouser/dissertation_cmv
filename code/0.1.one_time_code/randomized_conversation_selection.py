import pandas as pd

def select_random_conversations(csv_path, output_path, sample_size=2500, random_seed=42):
    # Load the dataset
    df = pd.read_csv(csv_path)

    # Randomly sample the conversations
    df_sampled = df.sample(n=sample_size, random_state=random_seed)

    # Save the sampled conversations
    df_sampled.to_csv(output_path, index=False)
    print(f"Saved {sample_size} randomly selected conversations to {output_path}")

if __name__ == "__main__":
    # Change this to point to your actual large CSV file
    input_csv_path = "/Users/ryanfunkhouser/Documents/Research/official_cmv_computational_small_stories/data/cleaned_final_convos.csv"
    output_csv_path = "../data/randomized_2500_conversations.csv"
    
    select_random_conversations(input_csv_path, output_csv_path)
