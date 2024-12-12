# Function to list available categories and their correct dataset names
def list_categories():
    categories = {
        "All Beauty": "raw_review_All_Beauty",
        "Toys and Games": "raw_review_Toys_and_Games",
        "Cell Phones and Accessories": "raw_review_Cell_Phones_and_Accessories",
        "Industrial and Scientific": "raw_review_Industrial_and_Scientific",
        "Gift Cards": "raw_review_Gift_Cards",
        "Musical Instruments": "raw_review_Musical_Instruments",
        "Electronics": "raw_review_Electronics",
        "Handmade Products": "raw_review_Handmade_Products",
        "Arts Crafts and Sewing": "raw_review_Arts_Crafts_and_Sewing",
        "Baby Products": "raw_review_Baby_Products",
        "Health and Household": "raw_review_Health_and_Household",
        "Office Products": "raw_review_Office_Products",
        "Digital Music": "raw_review_Digital_Music",
        "Grocery and Gourmet Food": "raw_review_Grocery_and_Gourmet_Food",
        "Sports and Outdoors": "raw_review_Sports_and_Outdoors",
        "Home and Kitchen": "raw_review_Home_and_Kitchen",
        "Subscription Boxes": "raw_review_Subscription_Boxes",
        "Tools and Home Improvement": "raw_review_Tools_and_Home_Improvement",
        "Pet Supplies": "raw_review_Pet_Supplies",
        "Video Games": "raw_review_Video_Games",
        "Kindle Store": "raw_review_Kindle_Store",
        "Clothing Shoes and Jewelry": "raw_review_Clothing_Shoes_and_Jewelry",
        "Patio Lawn and Garden": "raw_review_Patio_Lawn_and_Garden",
        "Books": "raw_review_Books",
        "Automotive": "raw_review_Automotive",
        "CDs and Vinyl": "raw_review_CDs_and_Vinyl",
        "Beauty and Personal Care": "raw_review_Beauty_and_Personal_Care",
        "Amazon Fashion": "raw_review_Amazon_Fashion",
        "Magazine Subscriptions": "raw_review_Magazine_Subscriptions",
        "Software": "raw_review_Software",
        "Health and Personal Care": "raw_review_Health_and_Personal_Care",
        "Appliances": "raw_review_Appliances",
        "Movies and TV": "raw_review_Movies_and_TV"
    }

    print("\nAvailable categories:")
    for idx, category in enumerate(categories.keys(), 1):
        print(f"{idx}. {category.replace('_', ' ')}")  # Replace underscores with spaces for better display
    return categories

# Function to load reviews with a manual progress bar
def load_reviews(dataset_name):
    print(f"\nDownloading raw review data for '{dataset_name}'...")

    # Load the dataset without immediately converting it to DataFrame
    dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", dataset_name, split="full", trust_remote_code=True)###.shuffle(seed=42).select(range(100000)) ## Sample the dataset HERE ######

    # Set up the manual progress bar for iterating through the dataset
    data = []
    with tqdm(total=len(dataset), desc="Downloading Review Data", unit="records") as pbar:
        for record in dataset:
            data.append(record)
            pbar.update(1)  # Update the progress bar for each record

    # Convert the collected records into a DataFrame
    df = pd.DataFrame(data)
    
    return df

# Function to load metadata with a manual progress bar
def load_metadata(dataset_name):
    print(f"\nDownloading raw metadata for '{dataset_name}'...")

    try:
        # Load the metadata dataset
        meta_dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", dataset_name.replace('raw_review', 'raw_meta'), split="full", trust_remote_code=True)

        # Set up the manual progress bar for iterating through the metadata
        meta_data = []
        with tqdm(total=len(meta_dataset), desc="Downloading Metadata", unit="records") as pbar:
            for record in meta_dataset:
                meta_data.append(record)
                pbar.update(1)  # Update the progress bar for each record

        # Convert the collected records into a DataFrame
        meta_df = pd.DataFrame(meta_data)
        print(f"Loaded {len(meta_df)} metadata records.")
    except Exception as e:
        print(f"Error while loading metadata: {e}")
        return None

    return meta_df

# Main function to run the CLI
# Main function to run the CLI
def download_data_by_category():
    print("Welcome to the Safety Dance Amazon Review Processor! (Data from McAuley Lab and CPSC)")

    # Step 1: List categories
    categories = list_categories()

    # Step 2: Prompt the user to select a category
    try:
        selection = int(input("\nSelect a category by number (e.g., 1 for All Beauty): "))
        if 1 <= selection <= len(categories):
            category = list(categories.keys())[selection - 1]  # Get the display name
            dataset_name = categories[category]  # Get the correct dataset name
            
            # Save the chosen category to a file for later use
            with open("chosen_category.txt", "w") as f:
                f.write(category)

            print(f"\nCategory '{category}' saved for later scripts.")
        else:
            print("Invalid selection. Please restart and choose a valid category.")
            return None, None
    except ValueError:
        print("Invalid input. Please enter a number corresponding to a category.")
        return None, None

    # Step 3: Download the review and metadata data
    df = load_reviews(dataset_name)

    # Check if reviews were successfully downloaded
    if df is None:
        print(f"Failed to download review data for '{category}'.")
        return None, None

    meta_df = load_metadata(dataset_name)

    # Check if metadata was successfully downloaded
    if meta_df is None:
        print(f"Failed to download metadata for '{category}'.")
        return None, None

    print(f"\nData download complete for '{category}' category!")
    return df, meta_df

# Ensure this function only runs when the script is executed directly
if __name__ == "__main__":
    df, meta_df = download_data_by_category()
