# AI-Based Disease Diagnosis and Dietary Recommendation System (Advanced Version)
# Required Libraries
library(shiny)
library(caret)
library(randomForest)
library(e1071)
library(DT)
library(ggplot2)
library(pdftools)
library(tesseract)
library(magick)
library(stringr)

# ============================================================================
# DATA PREPARATION - COMPREHENSIVE DISEASE DATABASE
# ============================================================================

create_disease_symptom_data <- function() {
  diseases <- c("Diabetes", "Hypertension", "Common Cold", "Influenza", 
                "Gastritis", "Migraine", "Asthma", "Arthritis",
                "Cancer", "Brain Stroke", "Cardiac Arrest", "Heart Attack",
                "Liver Damage", "Tuberculosis", "Paralysis", "Anemia",
                "Cataracts", "Glaucoma", "Macular Degeneration", "Conjunctivitis",
                "Eczema", "Psoriasis", "Melanoma")
  
  # Extended Symptom matrix (1 = present, 0 = absent)
  symptom_data <- data.frame(
    Fatigue = c(1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1),
    Headache = c(0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0),
    Fever = c(0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0),
    Cough = c(0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    Nausea = c(0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    Chest_Pain = c(0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    Joint_Pain = c(0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0),
    Frequent_Urination = c(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    Increased_Thirst = c(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    Dizziness = c(0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0),
    Abdominal_Pain = c(0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    Shortness_of_Breath = c(0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0),
    Sleeplessness = c(1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1),
    Tiredness = c(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1),
    Rapid_Hunger = c(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    Loss_of_Appetite = c(0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1),
    Blurred_Vision = c(1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0),
    Weight_Loss = c(1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1),
    Weakness = c(1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1),
    Numbness = c(0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0),
    Speech_Difficulty = c(0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0),
    Confusion = c(0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    Sweating = c(0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    Pale_Skin = c(0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0),
    Jaundice = c(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    Night_Sweats = c(0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    Swelling = c(0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0),
    Irregular_Heartbeat = c(0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    Eye_Pain = c(0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0),
    Eye_Redness = c(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0),
    Light_Sensitivity = c(0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0),
    Skin_Rash = c(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0),
    Itching = c(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0),
    Skin_Discoloration = c(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1),
    Skin_Lesions = c(0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1),
    Disease = diseases
  )
  
  return(symptom_data)
}

create_dietary_data <- function() {
  dietary_plans <- list(
    Diabetes = list(
      foods_to_eat = data.frame(
        Food = c("Steel-cut oatmeal", "Quinoa", "Brown rice", "Salmon fillet", 
                 "Chicken breast", "Spinach", "Broccoli", "Blueberries", 
                 "Almonds", "Greek yogurt", "Lentils", "Avocado"),
        Serving_Size = c("1 cup cooked", "1 cup cooked", "1 cup cooked", "4 oz", 
                         "4 oz", "2 cups raw", "1 cup cooked", "1 cup", 
                         "1 oz (23 nuts)", "1 cup", "1 cup cooked", "1/2 medium"),
        Calories = c(150, 222, 218, 206, 184, 14, 55, 84, 
                     164, 100, 230, 120),
        Protein_g = c(5, 8, 5, 23, 35, 2, 4, 1, 6, 17, 18, 2),
        Carbs_g = c(27, 39, 46, 0, 0, 2, 11, 21, 6, 6, 40, 6),
        Fat_g = c(3, 4, 2, 13, 4, 0, 1, 0.5, 14, 0, 1, 11),
        Fiber_g = c(4, 5, 4, 0, 0, 1, 5, 4, 4, 0, 16, 5)
      ),
      foods_to_avoid = c("Sugary drinks and sodas", "White bread", "Pastries", 
                         "Fried foods", "Processed meats", "Candy", "Fruit juice with added sugar"),
      sample_meal = data.frame(
        Meal = c("Breakfast", "Mid-Morning Snack", "Lunch", "Evening Snack", "Dinner"),
        Food = c("Steel-cut oatmeal with berries", "Apple with almond butter",
                 "Grilled chicken salad", "Handful of almonds",
                 "Baked salmon with quinoa and broccoli"),
        Quantity = c("1 cup + 1/2 cup berries", "1 medium + 2 tbsp",
                     "4 oz chicken + 3 cups salad", "1 oz",
                     "4 oz + 1 cup + 1 cup"),
        Calories = c(280, 180, 420, 170, 480),
        Protein_g = c(10, 5, 38, 7, 42),
        Carbs_g = c(48, 20, 32, 8, 38),
        Fat_g = c(8, 10, 18, 15, 22),
        Fiber_g = c(8, 4, 9, 4, 11)
      )
    ),
    
    Hypertension = list(
      foods_to_eat = data.frame(
        Food = c("Banana", "Spinach", "Salmon", "Oats", "Blueberries", 
                 "Garlic", "Greek yogurt", "Beets", "Pomegranate", "Olive oil",
                 "Dark chocolate", "Flaxseeds"),
        Serving_Size = c("1 medium", "2 cups raw", "4 oz", "1 cup cooked", "1 cup",
                         "3 cloves", "1 cup", "1 cup", "1/2 cup seeds", "1 tbsp",
                         "1 oz", "2 tbsp"),
        Calories = c(105, 14, 206, 150, 84, 13, 100, 58, 72, 119, 170, 75),
        Protein_g = c(1, 2, 23, 5, 1, 1, 17, 2, 1, 0, 2, 3),
        Carbs_g = c(27, 2, 0, 27, 21, 3, 6, 13, 16, 0, 13, 4),
        Fat_g = c(0, 0, 13, 3, 0.5, 0, 0, 0, 1, 14, 12, 6),
        Fiber_g = c(3, 1, 0, 4, 4, 0, 0, 4, 4, 0, 3, 4)
      ),
      foods_to_avoid = c("Table salt", "Processed foods", "Canned soups", 
                         "Pizza", "Pickles", "Alcohol", "Deli meats"),
      sample_meal = data.frame(
        Meal = c("Breakfast", "Mid-Morning", "Lunch", "Evening", "Dinner"),
        Food = c("Banana smoothie with oats", "Fresh berries",
                 "Mediterranean vegetable soup", "Dark chocolate",
                 "Grilled salmon with sweet potato"),
        Quantity = c("1 banana + 1/2 cup oats", "1 cup",
                     "2 cups", "2 squares (1 oz)",
                     "4 oz + 1 medium potato + 1 cup vegetables"),
        Calories = c(240, 90, 370, 100, 500),
        Protein_g = c(12, 1, 16, 2, 44),
        Carbs_g = c(42, 22, 54, 12, 42),
        Fat_g = c(5, 0.5, 10, 6, 24),
        Fiber_g = c(6, 5, 13, 2, 10)
      )
    ),
    
    Cancer = list(
      foods_to_eat = data.frame(
        Food = c("Broccoli", "Cauliflower", "Blueberries", "Salmon", "Turmeric",
                 "Green tea", "Walnuts", "Tomatoes", "Garlic", "Spinach",
                 "Flaxseeds", "Beans"),
        Serving_Size = c("1 cup cooked", "1 cup cooked", "1 cup", "4 oz", "1 tsp",
                         "1 cup", "1 oz", "1 medium", "3 cloves", "2 cups raw",
                         "2 tbsp", "1 cup cooked"),
        Calories = c(55, 29, 84, 206, 8, 2, 185, 22, 13, 14, 75, 225),
        Protein_g = c(4, 2, 1, 23, 0, 0, 4, 1, 1, 2, 3, 15),
        Carbs_g = c(11, 6, 21, 0, 2, 0, 4, 5, 3, 2, 4, 40),
        Fat_g = c(1, 0, 0.5, 13, 0, 0, 18, 0, 0, 0, 6, 1),
        Fiber_g = c(5, 3, 4, 0, 1, 0, 2, 1.5, 0, 1, 4, 15)
      ),
      foods_to_avoid = c("Processed meats", "Alcohol", "Refined sugars", 
                         "Trans fats", "Charred foods", "Excessive red meat"),
      sample_meal = data.frame(
        Meal = c("Breakfast", "Mid-Morning", "Lunch", "Evening", "Dinner"),
        Food = c("Green smoothie with berries", "Mixed nuts and green tea",
                 "Quinoa bowl with vegetables", "Carrot sticks",
                 "Baked fish with broccoli"),
        Quantity = c("2 cups + 1 cup berries", "1 oz nuts + 1 cup tea",
                     "1.5 cups + 2 cups vegetables", "1 cup",
                     "4 oz + 1 cup broccoli + 1 cup rice"),
        Calories = c(260, 200, 450, 80, 520),
        Protein_g = c(12, 8, 22, 2, 46),
        Carbs_g = c(38, 12, 62, 18, 48),
        Fat_g = c(10, 14, 15, 0.5, 20),
        Fiber_g = c(10, 4, 14, 4, 12)
      )
    ),
    
    "Liver Damage" = list(
      foods_to_eat = data.frame(
        Food = c("Coffee", "Oatmeal", "Walnuts", "Avocado", "Olive oil",
                 "Garlic", "Turmeric", "Beets", "Leafy greens", "Grapefruit",
                 "Green tea", "Salmon"),
        Serving_Size = c("1 cup", "1 cup cooked", "1 oz", "1/2 medium", "1 tbsp",
                         "3 cloves", "1 tsp", "1 cup", "2 cups", "1/2 fruit",
                         "1 cup", "4 oz"),
        Calories = c(2, 150, 185, 120, 119, 13, 8, 58, 14, 52, 2, 206),
        Protein_g = c(0, 5, 4, 2, 0, 1, 0, 2, 2, 1, 0, 23),
        Carbs_g = c(0, 27, 4, 6, 0, 3, 2, 13, 2, 13, 0, 0),
        Fat_g = c(0, 3, 18, 11, 14, 0, 0, 0, 0, 0, 0, 13),
        Fiber_g = c(0, 4, 2, 5, 0, 0, 1, 4, 1, 2, 0, 0)
      ),
      foods_to_avoid = c("Alcohol (strictly)", "Fatty foods", "Fried foods", 
                         "Processed foods", "Excessive salt", "Red meat"),
      sample_meal = data.frame(
        Meal = c("Breakfast", "Mid-Morning", "Lunch", "Evening", "Dinner"),
        Food = c("Oatmeal with walnuts", "Coffee and apple",
                 "Baked fish with vegetables", "Green tea",
                 "Grilled chicken with brown rice"),
        Quantity = c("1 cup + 1 oz nuts", "1 cup + 1 apple",
                     "4 oz + 2 cups vegetables", "1 cup",
                     "4 oz + 1 cup rice + salad"),
        Calories = c(260, 110, 410, 70, 470),
        Protein_g = c(9, 1, 38, 1, 42),
        Carbs_g = c(46, 26, 35, 16, 50),
        Fat_g = c(8, 1, 16, 0.5, 16),
        Fiber_g = c(8, 4, 8, 3, 9)
      )
    ),
    
    Cataracts = list(
      foods_to_eat = data.frame(
        Food = c("Carrots", "Sweet potato", "Spinach", "Kale", "Salmon",
                 "Blueberries", "Oranges", "Eggs", "Sunflower seeds", "Almonds",
                 "Bell peppers", "Broccoli"),
        Serving_Size = c("1 cup", "1 medium", "2 cups raw", "1 cup cooked", "4 oz",
                         "1 cup", "1 medium", "1 large", "1 oz", "1 oz",
                         "1 cup", "1 cup cooked"),
        Calories = c(52, 103, 14, 33, 206, 84, 62, 72, 165, 164, 37, 55),
        Protein_g = c(1, 2, 2, 3, 23, 1, 1, 6, 6, 6, 1, 4),
        Carbs_g = c(12, 24, 2, 7, 0, 21, 15, 1, 6, 6, 9, 11),
        Fat_g = c(0, 0, 0, 1, 13, 0.5, 0, 5, 14, 14, 0, 1),
        Fiber_g = c(4, 4, 1, 1, 0, 4, 3, 0, 3, 4, 3, 5)
      ),
      foods_to_avoid = c("Excessive alcohol", "Smoking-related foods", "High-sodium foods",
                         "Processed sugars", "Trans fats"),
      sample_meal = data.frame(
        Meal = c("Breakfast", "Mid-Morning", "Lunch", "Evening", "Dinner"),
        Food = c("Scrambled eggs with spinach", "Orange with almonds",
                 "Grilled salmon salad", "Carrot sticks",
                 "Baked chicken with sweet potato"),
        Quantity = c("2 eggs + 2 cups spinach", "1 orange + 1 oz almonds",
                     "4 oz + 3 cups salad", "1 cup",
                     "4 oz + 1 medium potato + 1 cup broccoli"),
        Calories = c(230, 230, 440, 52, 510),
        Protein_g = c(16, 7, 42, 1, 46),
        Carbs_g = c(8, 21, 28, 12, 46),
        Fat_g = c(16, 14, 20, 0, 18),
        Fiber_g = c(2, 7, 10, 4, 11)
      )
    ),
    
    Eczema = list(
      foods_to_eat = data.frame(
        Food = c("Fatty fish", "Flaxseeds", "Walnuts", "Probiotic yogurt", "Spinach",
                 "Broccoli", "Sweet potato", "Berries", "Pumpkin seeds", "Olive oil",
                 "Avocado", "Green tea"),
        Serving_Size = c("4 oz", "2 tbsp", "1 oz", "1 cup", "2 cups",
                         "1 cup cooked", "1 medium", "1 cup", "1 oz", "1 tbsp",
                         "1/2 medium", "1 cup"),
        Calories = c(206, 75, 185, 100, 14, 55, 103, 84, 150, 119, 120, 2),
        Protein_g = c(23, 3, 4, 17, 2, 4, 2, 1, 7, 0, 2, 0),
        Carbs_g = c(0, 4, 4, 6, 2, 11, 24, 21, 5, 0, 6, 0),
        Fat_g = c(13, 6, 18, 0, 0, 1, 0, 0.5, 13, 14, 11, 0),
        Fiber_g = c(0, 4, 2, 0, 1, 5, 4, 4, 2, 0, 5, 0)
      ),
      foods_to_avoid = c("Dairy (if sensitive)", "Eggs (if allergic)", "Nuts (if allergic)",
                         "Gluten (if sensitive)", "Processed foods", "Citrus (if trigger)"),
      sample_meal = data.frame(
        Meal = c("Breakfast", "Mid-Morning", "Lunch", "Evening", "Dinner"),
        Food = c("Oatmeal with berries and flaxseeds", "Probiotic yogurt",
                 "Grilled salmon with quinoa", "Pumpkin seeds",
                 "Baked chicken with sweet potato"),
        Quantity = c("1 cup + 1 cup berries + 2 tbsp", "1 cup",
                     "4 oz + 1 cup quinoa + salad", "1 oz",
                     "4 oz + 1 potato + 2 cups vegetables"),
        Calories = c(310, 100, 470, 150, 490),
        Protein_g = c(9, 17, 40, 7, 44),
        Carbs_g = c(56, 6, 42, 5, 52),
        Fat_g = c(9, 0, 22, 13, 16),
        Fiber_g = c(12, 0, 10, 2, 11)
      )
    ),
    
    
    Influenza = list(
      foods_to_eat = data.frame(
        Food = c("Chicken broth", "Oranges", "Ginger", "Honey", "Garlic",
                 "Green tea", "Yogurt", "Spinach", "Almonds", "Sweet potato",
                 "Eggs", "Turmeric"),
        Serving_Size = c("1 cup", "1 medium", "1 inch piece", "1 tbsp", "3 cloves",
                         "1 cup", "1 cup", "2 cups", "1 oz", "1 medium",
                         "1 large", "1 tsp"),
        Calories = c(38, 62, 2, 64, 13, 2, 100, 14, 164, 103, 72, 8),
        Protein_g = c(5, 1, 0, 0, 1, 0, 17, 2, 6, 2, 6, 0),
        Carbs_g = c(1, 15, 0.5, 17, 3, 0, 6, 2, 6, 24, 1, 2),
        Fat_g = c(1, 0, 0, 0, 0, 0, 0, 0, 14, 0, 5, 0),
        Fiber_g = c(0, 3, 0, 0, 0, 0, 0, 1, 4, 4, 0, 1)
      ),
      foods_to_avoid = c("Sugary foods", "Processed foods", "Alcohol", "Fried foods",
                         "Caffeinated beverages", "Dairy (if mucus increases)"),
      sample_meal = data.frame(
        Meal = c("Breakfast", "Mid-Morning", "Lunch", "Evening", "Dinner"),
        Food = c("Scrambled eggs with spinach", "Orange and green tea",
                 "Chicken broth soup with vegetables", "Ginger tea with honey",
                 "Baked sweet potato with yogurt"),
        Quantity = c("2 eggs + 2 cups spinach", "1 orange + 1 cup tea",
                     "2 cups soup", "1 cup tea + 1 tbsp honey",
                     "1 potato + 1 cup yogurt"),
        Calories = c(158, 64, 320, 66, 203),
        Protein_g = c(14, 1, 28, 0, 19),
        Carbs_g = c(5, 15, 24, 17, 30),
        Fat_g = c(10, 0, 8, 0, 0),
        Fiber_g = c(2, 3, 6, 0, 4)
      )
    ),
    
    Migraine = list(
      foods_to_eat = data.frame(
        Food = c("Salmon", "Spinach", "Quinoa", "Almonds", "Bananas",
                 "Watermelon", "Ginger", "Brown rice", "Sweet potato", "Blueberries",
                 "Eggs", "Olive oil"),
        Serving_Size = c("4 oz", "2 cups", "1 cup cooked", "1 oz", "1 medium",
                         "2 cups", "1 inch", "1 cup cooked", "1 medium", "1 cup",
                         "1 large", "1 tbsp"),
        Calories = c(206, 14, 222, 164, 105, 92, 2, 218, 103, 84, 72, 119),
        Protein_g = c(23, 2, 8, 6, 1, 2, 0, 5, 2, 1, 6, 0),
        Carbs_g = c(0, 2, 39, 6, 27, 23, 0.5, 46, 24, 21, 1, 0),
        Fat_g = c(13, 0, 4, 14, 0, 0, 0, 2, 0, 0.5, 5, 14),
        Fiber_g = c(0, 1, 5, 4, 3, 1, 0, 4, 4, 4, 0, 0)
      ),
      foods_to_avoid = c("Aged cheese", "Processed meats", "Alcohol (especially red wine)", 
                         "Chocolate", "Artificial sweeteners", "MSG", "Caffeine (excessive)"),
      sample_meal = data.frame(
        Meal = c("Breakfast", "Mid-Morning", "Lunch", "Evening", "Dinner"),
        Food = c("Scrambled eggs with spinach", "Banana with almonds",
                 "Quinoa bowl with salmon", "Watermelon slices",
                 "Brown rice with sweet potato and vegetables"),
        Quantity = c("2 eggs + 2 cups spinach", "1 banana + 1 oz almonds",
                     "1 cup quinoa + 4 oz salmon", "2 cups",
                     "1 cup rice + 1 potato + 2 cups veggies"),
        Calories = c(158, 269, 428, 92, 435),
        Protein_g = c(14, 7, 31, 2, 9),
        Carbs_g = c(5, 33, 39, 23, 94),
        Fat_g = c(10, 14, 17, 0, 2),
        Fiber_g = c(2, 7, 5, 1, 12)
      )
    ),
    
    Asthma = list(
      foods_to_eat = data.frame(
        Food = c("Salmon", "Spinach", "Apples", "Bananas", "Carrots",
                 "Tomatoes", "Broccoli", "Sweet potato", "Berries", "Avocado",
                 "Flaxseeds", "Walnuts"),
        Serving_Size = c("4 oz", "2 cups", "1 medium", "1 medium", "1 cup",
                         "1 medium", "1 cup", "1 medium", "1 cup", "1/2 medium",
                         "2 tbsp", "1 oz"),
        Calories = c(206, 14, 95, 105, 52, 22, 55, 103, 84, 120, 75, 185),
        Protein_g = c(23, 2, 0.5, 1, 1, 1, 4, 2, 1, 2, 3, 4),
        Carbs_g = c(0, 2, 25, 27, 12, 5, 11, 24, 21, 6, 4, 4),
        Fat_g = c(13, 0, 0, 0, 0, 0, 1, 0, 0.5, 11, 6, 18),
        Fiber_g = c(0, 1, 4, 3, 4, 1.5, 5, 4, 4, 5, 4, 2)
      ),
      foods_to_avoid = c("Sulfites (dried fruits, wine)", "Dairy (if allergic)", "Shellfish",
                         "Eggs (if allergic)", "Peanuts (if allergic)", "Processed foods"),
      sample_meal = data.frame(
        Meal = c("Breakfast", "Mid-Morning", "Lunch", "Evening", "Dinner"),
        Food = c("Oatmeal with berries and flaxseeds", "Apple with almond butter",
                 "Grilled salmon with broccoli", "Carrot and celery sticks",
                 "Baked chicken with sweet potato"),
        Quantity = c("1 cup + 1 cup berries + 2 tbsp", "1 apple + 2 tbsp butter",
                     "4 oz + 1 cup broccoli + quinoa", "1 cup each",
                     "4 oz + 1 potato + salad"),
        Calories = c(309, 285, 483, 93, 470),
        Protein_g = c(9, 5, 37, 2, 42),
        Carbs_g = c(56, 31, 40, 22, 50),
        Fat_g = c(9, 16, 19, 0.5, 16),
        Fiber_g = c(12, 8, 10, 8, 11)
      )
    ),
    
    Arthritis = list(
      foods_to_eat = data.frame(
        Food = c("Salmon", "Turmeric", "Ginger", "Olive oil", "Spinach",
                 "Broccoli", "Walnuts", "Berries", "Cherries", "Oranges",
                 "Green tea", "Garlic"),
        Serving_Size = c("4 oz", "1 tsp", "1 inch", "1 tbsp", "2 cups",
                         "1 cup", "1 oz", "1 cup", "1 cup", "1 medium",
                         "1 cup", "3 cloves"),
        Calories = c(206, 8, 2, 119, 14, 55, 185, 84, 97, 62, 2, 13),
        Protein_g = c(23, 0, 0, 0, 2, 4, 4, 1, 2, 1, 0, 1),
        Carbs_g = c(0, 2, 0.5, 0, 2, 11, 4, 21, 24, 15, 0, 3),
        Fat_g = c(13, 0, 0, 14, 0, 1, 18, 0.5, 0, 0, 0, 0),
        Fiber_g = c(0, 1, 0, 0, 1, 5, 2, 4, 3, 3, 0, 0)
      ),
      foods_to_avoid = c("Red meat", "Processed foods", "Refined carbs", "Fried foods",
                         "Alcohol", "Sugar and sweets", "Trans fats"),
      sample_meal = data.frame(
        Meal = c("Breakfast", "Mid-Morning", "Lunch", "Evening", "Dinner"),
        Food = c("Oatmeal with berries and walnuts", "Orange and green tea",
                 "Grilled salmon with spinach salad", "Cherries",
                 "Stir-fry with turmeric and ginger"),
        Quantity = c("1 cup + 1 cup berries + 1 oz nuts", "1 orange + 1 cup tea",
                     "4 oz + 3 cups spinach + olive oil", "1 cup",
                     "Mixed vegetables + 4 oz chicken + spices"),
        Calories = c(419, 64, 352, 97, 450),
        Protein_g = c(9, 1, 35, 2, 40),
        Carbs_g = c(52, 15, 8, 24, 38),
        Fat_g = c(21, 0, 28, 0, 20),
        Fiber_g = c(10, 3, 6, 3, 10)
      )
    ),
    
                     "Macular Degeneration" = list(
                       foods_to_eat = data.frame(
                         Food = c("Salmon", "Spinach", "Kale", "Eggs", "Oranges",
                                  "Blueberries", "Almonds", "Sunflower seeds", "Sweet potato", "Bell peppers",
                                  "Carrots", "Broccoli"),
                         Serving_Size = c("4 oz", "2 cups", "1 cup cooked", "1 large", "1 medium",
                                          "1 cup", "1 oz", "1 oz", "1 medium", "1 cup",
                                          "1 cup", "1 cup cooked"),
                         Calories = c(206, 14, 33, 72, 62, 84, 164, 165, 103, 37, 52, 55),
                         Protein_g = c(23, 2, 3, 6, 1, 1, 6, 6, 2, 1, 1, 4),
                         Carbs_g = c(0, 2, 7, 1, 15, 21, 6, 6, 24, 9, 12, 11),
                         Fat_g = c(13, 0, 1, 5, 0, 0.5, 14, 14, 0, 0, 0, 1),
                         Fiber_g = c(0, 1, 1, 0, 3, 4, 4, 3, 4, 3, 4, 5)
                       ),
                       foods_to_avoid = c("Trans fats", "Saturated fats", "Refined carbohydrates", "Processed foods",
                                          "High-sodium foods", "Sugary drinks"),
                       sample_meal = data.frame(
                         Meal = c("Breakfast", "Mid-Morning", "Lunch", "Evening", "Dinner"),
                         Food = c("Scrambled eggs with kale", "Orange with almonds",
                                  "Grilled salmon with spinach salad", "Carrot and bell pepper sticks",
                                  "Chicken with sweet potato and broccoli"),
                         Quantity = c("2 eggs + 1 cup kale", "1 orange + 1 oz almonds",
                                      "4 oz + 3 cups spinach + olive oil", "1 cup each",
                                      "4 oz + 1 potato + 1 cup broccoli"),
                         Calories = c(177, 226, 353, 89, 526),
                         Protein_g = c(15, 7, 35, 2, 50),
                         Carbs_g = c(8, 21, 8, 21, 59),
                         Fat_g = c(11, 14, 28, 0, 18),
                         Fiber_g = c(1, 7, 6, 7, 13)
                       )
                     ),
                     
                     Conjunctivitis = list(
                       foods_to_eat = data.frame(
                         Food = c("Spinach", "Carrots", "Sweet potato", "Bell peppers", "Berries",
                                  "Salmon", "Eggs", "Almonds", "Sunflower seeds", "Oranges",
                                  "Tomatoes", "Broccoli"),
                         Serving_Size = c("2 cups", "1 cup", "1 medium", "1 cup", "1 cup",
                                          "4 oz", "1 large", "1 oz", "1 oz", "1 medium",
                                          "1 medium", "1 cup cooked"),
                         Calories = c(14, 52, 103, 37, 84, 206, 72, 164, 165, 62, 22, 55),
                         Protein_g = c(2, 1, 2, 1, 1, 23, 6, 6, 6, 1, 1, 4),
                         Carbs_g = c(2, 12, 24, 9, 21, 0, 1, 6, 6, 15, 5, 11),
                         Fat_g = c(0, 0, 0, 0, 0.5, 13, 5, 14, 14, 0, 0, 1),
                         Fiber_g = c(1, 4, 4, 3, 4, 0, 0, 4, 3, 3, 1.5, 5)
                       ),
                       foods_to_avoid = c("Processed foods", "High-sodium foods", "Sugary foods", "Alcohol",
                                          "Fried foods", "Trans fats"),
                       sample_meal = data.frame(
                         Meal = c("Breakfast", "Mid-Morning", "Lunch", "Evening", "Dinner"),
                         Food = c("Scrambled eggs with spinach", "Berries and almonds",
                                  "Grilled salmon with vegetables", "Carrot sticks",
                                  "Chicken with sweet potato and broccoli"),
                         Quantity = c("2 eggs + 2 cups spinach", "1 cup + 1 oz",
                                      "4 oz + 2 cups mixed vegetables", "1 cup",
                                      "4 oz + 1 potato + 1 cup broccoli"),
                         Calories = c(158, 248, 445, 52, 526),
                         Protein_g = c(14, 7, 42, 1, 50),
                         Carbs_g = c(5, 27, 26, 12, 59),
                         Fat_g = c(10, 14.5, 19, 0, 18),
                         Fiber_g = c(2, 8, 12, 4, 13)
                       )
                     ),
                     
                     Melanoma = list(
                       foods_to_eat = data.frame(
                         Food = c("Berries", "Leafy greens", "Tomatoes", "Carrots", "Sweet potato",
                                  "Salmon", "Walnuts", "Green tea", "Turmeric", "Garlic",
                                  "Broccoli", "Citrus fruits"),
                         Serving_Size = c("1 cup", "2 cups", "1 medium", "1 cup", "1 medium",
                                          "4 oz", "1 oz", "1 cup", "1 tsp", "3 cloves",
                                          "1 cup cooked", "1 medium"),
                         Calories = c(84, 14, 22, 52, 103, 206, 185, 2, 8, 13, 55, 62),
                         Protein_g = c(1, 2, 1, 1, 2, 23, 4, 0, 0, 1, 4, 1),
                         Carbs_g = c(21, 2, 5, 12, 24, 0, 4, 0, 2, 3, 11, 15),
                         Fat_g = c(0.5, 0, 0, 0, 0, 13, 18, 0, 0, 0, 1, 0),
                         Fiber_g = c(4, 1, 1.5, 4, 4, 0, 2, 0, 1, 0, 5, 3)
                       ),
                       foods_to_avoid = c("Processed meats", "Fried foods", "Refined sugars", "Trans fats",
                                          "Excessive alcohol", "Charred/burnt foods"),
                       sample_meal = data.frame(
                         Meal = c("Breakfast", "Mid-Morning", "Lunch", "Evening", "Dinner"),
                         Food = c("Berry smoothie with greens", "Orange and walnuts",
                                  "Grilled salmon with tomato salad", "Carrot sticks with hummus",
                                  "Stir-fry with turmeric and vegetables"),
                         Quantity = c("2 cups berries + 2 cups greens", "1 orange + 1 oz nuts",
                                      "4 oz + 2 tomatoes + greens", "1 cup + 1/2 cup",
                                      "Mixed vegetables + 4 oz chicken + spices"),
                         Calories = c(182, 247, 456, 200, 460),
                         Protein_g = c(6, 5, 44, 8, 42),
                         Carbs_g = c(44, 19, 22, 28, 38),
                         Fat_g = c(1, 18, 20, 8, 20),
                         Fiber_g = c(10, 5, 10, 8, 12)
                       )
                     ),
                     
                     "Brain Stroke" = list(
                       foods_to_eat = data.frame(
                         Food = c("Salmon", "Walnuts", "Blueberries", "Spinach", "Oats",
                                  "Olive oil", "Avocado", "Beans", "Tomatoes", "Dark chocolate",
                                  "Green tea", "Flaxseeds"),
                         Serving_Size = c("4 oz", "1 oz", "1 cup", "2 cups", "1 cup cooked",
                                          "1 tbsp", "1/2 medium", "1 cup cooked", "1 medium", "1 oz",
                                          "1 cup", "2 tbsp"),
                         Calories = c(206, 185, 84, 14, 150, 119, 120, 225, 22, 170, 2, 75),
                         Protein_g = c(23, 4, 1, 2, 5, 0, 2, 15, 1, 2, 0, 3),
                         Carbs_g = c(0, 4, 21, 2, 27, 0, 6, 40, 5, 13, 0, 4),
                         Fat_g = c(13, 18, 0.5, 0, 3, 14, 11, 1, 0, 12, 0, 6),
                         Fiber_g = c(0, 2, 4, 1, 4, 0, 5, 15, 1.5, 3, 0, 4)
                       ),
                       foods_to_avoid = c("High sodium foods", "Trans fats", "Processed meats", "Red meat",
                                          "Sugary drinks", "Fried foods", "Excessive alcohol"),
                       sample_meal = data.frame(
                         Meal = c("Breakfast", "Mid-Morning", "Lunch", "Evening", "Dinner"),
                         Food = c("Oatmeal with berries and walnuts", "Apple with almond butter",
                                  "Mediterranean salad with salmon", "Dark chocolate and green tea",
                                  "Bean stew with vegetables"),
                         Quantity = c("1 cup + 1 cup berries + 1 oz nuts", "1 apple + 2 tbsp",
                                      "4 oz salmon + mixed greens + olive oil", "1 oz + 1 cup",
                                      "1.5 cups beans + 2 cups vegetables"),
                         Calories = c(419, 285, 450, 172, 480),
                         Protein_g = c(10, 5, 40, 2, 28),
                         Carbs_g = c(52, 31, 28, 13, 75),
                         Fat_g = c(21, 16, 26, 12, 8),
                         Fiber_g = c(10, 8, 11, 3, 24)
                       )
                     ),
                     
                     "Cardiac Arrest" = list(
                       foods_to_eat = data.frame(
                         Food = c("Oily fish", "Oats", "Nuts", "Berries", "Leafy greens",
                                  "Avocados", "Olive oil", "Legumes", "Tomatoes", "Flaxseeds",
                                  "Chia seeds", "Green tea"),
                         Serving_Size = c("4 oz", "1 cup cooked", "1 oz", "1 cup", "2 cups",
                                          "1/2 medium", "1 tbsp", "1 cup cooked", "1 medium", "2 tbsp",
                                          "2 tbsp", "1 cup"),
                         Calories = c(206, 150, 164, 84, 14, 120, 119, 225, 22, 75, 138, 2),
                         Protein_g = c(23, 5, 6, 1, 2, 2, 0, 15, 1, 3, 5, 0),
                         Carbs_g = c(0, 27, 6, 21, 2, 6, 0, 40, 5, 4, 12, 0),
                         Fat_g = c(13, 3, 14, 0.5, 0, 11, 14, 1, 0, 6, 9, 0),
                         Fiber_g = c(0, 4, 4, 4, 1, 5, 0, 15, 1.5, 4, 10, 0)
                       ),
                       foods_to_avoid = c("Saturated fats", "Trans fats", "Processed meats", "Fried foods",
                                          "Excessive salt", "Refined carbs", "Sugary foods"),
                       sample_meal = data.frame(
                         Meal = c("Breakfast", "Mid-Morning", "Lunch", "Evening", "Dinner"),
                         Food = c("Oats with berries and chia seeds", "Green tea with almonds",
                                  "Grilled fish with bean salad", "Carrot sticks with hummus",
                                  "Baked salmon with vegetables"),
                         Quantity = c("1 cup + 1 cup berries + 2 tbsp", "1 cup + 1 oz",
                                      "4 oz + 1 cup beans + greens", "1 cup + 1/2 cup",
                                      "4 oz + 2 cups vegetables + olive oil"),
                         Calories = c(372, 166, 465, 200, 480),
                         Protein_g = c(11, 6, 43, 8, 45),
                         Carbs_g = c(60, 6, 42, 28, 24),
                         Fat_g = c(12.5, 14, 20, 8, 28),
                         Fiber_g = c(18, 4, 16, 8, 7)
                       )
                     ),
                     
                     "Heart Attack" = list(
                       foods_to_eat = data.frame(
                         Food = c("Salmon", "Sardines", "Steel-cut oats", "Almonds", "Walnuts",
                                  "Berries", "Spinach", "Avocado", "Olive oil", "Beans",
                                  "Tomatoes", "Flaxseeds"),
                         Serving_Size = c("4 oz", "3.75 oz", "1 cup cooked", "1 oz", "1 oz",
                                          "1 cup", "2 cups", "1/2 medium", "1 tbsp", "1 cup cooked",
                                          "1 medium", "2 tbsp"),
                         Calories = c(206, 208, 150, 164, 185, 84, 14, 120, 119, 225, 22, 75),
                         Protein_g = c(23, 25, 5, 6, 4, 1, 2, 2, 0, 15, 1, 3),
                         Carbs_g = c(0, 0, 27, 6, 4, 21, 2, 6, 0, 40, 5, 4),
                         Fat_g = c(13, 11, 3, 14, 18, 0.5, 0, 11, 14, 1, 0, 6),
                         Fiber_g = c(0, 0, 4, 4, 2, 4, 1, 5, 0, 15, 1.5, 4)
                       ),
                       foods_to_avoid = c("Saturated fats", "Trans fats", "Processed meats", "Fried foods",
                                          "Excessive salt", "Refined carbs", "Sugary drinks", "Full-fat dairy"),
                       sample_meal = data.frame(
                         Meal = c("Breakfast", "Mid-Morning", "Lunch", "Evening", "Dinner"),
                         Food = c("Steel-cut oats with berries and flaxseeds", "Green tea with almonds",
                                  "Grilled chicken with bean salad", "Carrot sticks with hummus",
                                  "Baked salmon with steamed vegetables"),
                         Quantity = c("1 cup + 1 cup berries + 2 tbsp", "1 cup + 1 oz",
                                      "4 oz + 1 cup beans + greens", "1 cup + 1/2 cup",
                                      "4 oz + 2 cups vegetables + olive oil"),
                         Calories = c(309, 166, 460, 200, 480),
                         Protein_g = c(9, 6, 40, 8, 45),
                         Carbs_g = c(52, 6, 42, 28, 24),
                         Fat_g = c(9.5, 14, 20, 8, 28),
                         Fiber_g = c(12, 4, 16, 8, 7)
                       )
                     ),
                     
                     Paralysis = list(
                       foods_to_eat = data.frame(
                         Food = c("Salmon", "Spinach", "Blueberries", "Walnuts", "Eggs",
                                  "Whole grains", "Avocado", "Beans", "Olive oil", "Greek yogurt",
                                  "Sweet potato", "Broccoli"),
                         Serving_Size = c("4 oz", "2 cups", "1 cup", "1 oz", "1 large",
                                          "1 cup cooked", "1/2 medium", "1 cup cooked", "1 tbsp", "1 cup",
                                          "1 medium", "1 cup cooked"),
                         Calories = c(206, 14, 84, 185, 72, 218, 120, 225, 119, 100, 103, 55),
                         Protein_g = c(23, 2, 1, 4, 6, 5, 2, 15, 0, 17, 2, 4),
                         Carbs_g = c(0, 2, 21, 4, 1, 46, 6, 40, 0, 6, 24, 11),
                         Fat_g = c(13, 0, 0.5, 18, 5, 2, 11, 1, 14, 0, 0, 1),
                         Fiber_g = c(0, 1, 4, 2, 0, 4, 5, 15, 0, 0, 4, 5)
                       ),
                       foods_to_avoid = c("High sodium foods", "Saturated fats", "Trans fats", "Processed foods",
                                          "Excessive sugar", "Alcohol"),
                       sample_meal = data.frame(
                         Meal = c("Breakfast", "Mid-Morning", "Lunch", "Evening", "Dinner"),
                         Food = c("Scrambled eggs with spinach and toast", "Berries and yogurt",
                                  "Grilled salmon with quinoa salad", "Greek yogurt",
                                  "Chicken with sweet potato and broccoli"),
                         Quantity = c("2 eggs + 2 cups spinach + 1 slice", "1 cup + 1 cup",
                                      "4 oz + 1 cup quinoa + greens", "1 cup",
                                      "4 oz + 1 potato + 1 cup broccoli"),
                         Calories = c(300, 184, 460, 100, 526),
                         Protein_g = c(18, 18, 42, 17, 50),
                         Carbs_g = c(25, 27, 42, 6, 59),
                         Fat_g = c(14, 0.5, 20, 0, 18),
                         Fiber_g = c(4, 4, 10, 0, 13)
                       )
                     ),
                     
                     Anemia = list(
                       foods_to_eat = data.frame(
                         Food = c("Red meat (lean)", "Chicken liver", "Spinach", "Lentils", "Beans",
                                  "Iron-fortified cereals", "Eggs", "Dried apricots", "Pumpkin seeds", "Beetroot",
                                  "Quinoa", "Tofu"),
                         Serving_Size = c("4 oz", "3 oz", "2 cups cooked", "1 cup cooked", "1 cup cooked",
                                          "1 cup", "1 large", "1/2 cup", "1 oz", "1 cup",
                                          "1 cup cooked", "4 oz"),
                         Calories = c(280, 172, 41, 230, 225, 120, 72, 157, 150, 58, 222, 86),
                         Protein_g = c(32, 26, 5, 18, 15, 3, 6, 2, 7, 2, 8, 10),
                         Carbs_g = c(0, 1, 7, 40, 40, 27, 1, 41, 5, 13, 39, 2),
                         Fat_g = c(16, 6, 0, 1, 1, 1, 5, 0, 13, 0, 4, 5),
                         Fiber_g = c(0, 0, 4, 16, 15, 3, 0, 5, 2, 4, 5, 2)
                       ),
                       foods_to_avoid = c("Coffee and tea (with meals)", "Calcium-rich foods (with iron meals)",
                                          "Whole grains (with iron meals)", "Alcohol", "Processed foods"),
                       sample_meal = data.frame(
                         Meal = c("Breakfast", "Mid-Morning", "Lunch", "Evening", "Dinner"),
                         Food = c("Iron-fortified cereal with orange juice", "Dried apricots and nuts",
                                  "Lean beef with spinach salad", "Beetroot juice",
                                  "Chicken liver with lentils and vegetables"),
                         Quantity = c("1 cup cereal + 1 cup OJ", "1/2 cup + 1 oz",
                                      "4 oz + 2 cups spinach cooked", "1 cup",
                                      "3 oz + 1 cup lentils + 2 cups veggies"),
                         Calories = c(290, 320, 321, 58, 520),
                         Protein_g = c(4, 9, 37, 2, 52),
                         Carbs_g = c(60, 46, 7, 13, 54),
                         Fat_g = c(1.5, 14, 16, 0, 8),
                         Fiber_g = c(3, 9, 4, 4, 20)
                       )
                     ),
                     
                     Glaucoma = list(
                       foods_to_eat = data.frame(
                         Food = c("Leafy greens", "Salmon", "Eggs", "Carrots", "Berries",
                                  "Nuts", "Citrus fruits", "Avocado", "Sweet potato", "Tomatoes",
                                  "Broccoli", "Bell peppers"),
                         Serving_Size = c("2 cups", "4 oz", "1 large", "1 cup", "1 cup",
                                          "1 oz", "1 medium", "1/2 medium", "1 medium", "1 medium",
                                          "1 cup", "1 cup"),
                         Calories = c(14, 206, 72, 52, 84, 164, 62, 120, 103, 22, 55, 37),
                         Protein_g = c(2, 23, 6, 1, 1, 6, 1, 2, 2, 1, 4, 1),
                         Carbs_g = c(2, 0, 1, 12, 21, 6, 15, 6, 24, 5, 11, 9),
                         Fat_g = c(0, 13, 5, 0, 0.5, 14, 0, 11, 0, 0, 1, 0),
                         Fiber_g = c(1, 0, 0, 4, 4, 4, 3, 5, 4, 1.5, 5, 3)
                       ),
                       foods_to_avoid = c("Excessive caffeine", "Trans fats", "Saturated fats", "High sodium foods",
                                          "Sugary drinks", "Processed foods"),
                       sample_meal = data.frame(
                         Meal = c("Breakfast", "Mid-Morning", "Lunch", "Evening", "Dinner"),
                         Food = c("Scrambled eggs with spinach", "Berries and almonds",
                                  "Grilled salmon with mixed vegetables", "Carrot and bell pepper sticks",
                                  "Chicken with sweet potato and broccoli"),
                         Quantity = c("2 eggs + 2 cups greens", "1 cup + 1 oz",
                                      "4 oz + 2 cups vegetables", "1 cup each",
                                      "4 oz + 1 potato + 1 cup broccoli"),
                         Calories = c(158, 248, 456, 89, 526),
                         Protein_g = c(14, 7, 44, 2, 50),
                         Carbs_g = c(5, 27, 28, 21, 59),
                         Fat_g = c(10, 14.5, 20, 0, 18),
                         Fiber_g = c(2, 8, 14, 8.5, 13)
                       )
                     ),	
    
    Tuberculosis = list(
      foods_to_eat = data.frame(
        Food = c("Eggs", "Chicken breast", "Lentils", "Milk", "Yogurt",
                 "Almonds", "Spinach", "Sweet potato", "Orange", "Banana",
                 "Brown rice", "Salmon"),
        Serving_Size = c("2 large", "4 oz", "1 cup cooked", "1 cup", "1 cup",
                         "1 oz", "2 cups", "1 medium", "1 medium", "1 medium",
                         "1 cup cooked", "4 oz"),
        Calories = c(144, 187, 230, 149, 149,
                     164, 14, 103, 62, 105,
                     216, 206),
        Protein_g = c(12, 35, 18, 8, 8,
                      6, 2, 2, 1, 1,
                      5, 23),
        Carbs_g = c(2, 0, 40, 12, 11,
                    6, 2, 24, 15, 27,
                    45, 0),
        Fat_g = c(10, 4, 1, 8, 8,
                  14, 0, 0, 0, 0,
                  2, 13),
        Fiber_g = c(0, 0, 16, 0, 0,
                    4, 1, 4, 3, 3,
                    4, 0)
      ),
      foods_to_avoid = c("Alcohol", "Refined sugar", "White bread", "Fried foods",
                         "Processed meats", "Caffeinated beverages", "High-fat junk food"),
      sample_meal = data.frame(
        Meal = c("Breakfast", "Mid-Morning", "Lunch", "Evening", "Dinner"),
        Food = c("Scrambled eggs with spinach and milk", "Banana with almonds",
                 "Grilled chicken with brown rice and vegetables", "Orange and yogurt",
                 "Baked salmon with sweet potato and lentils"),
        Quantity = c("2 eggs + 1 cup milk + 1 cup spinach", "1 banana + 1 oz almonds",
                     "4 oz chicken + 1 cup rice + 1 cup vegetables", "1 orange + 1 cup yogurt",
                     "4 oz salmon + 1 potato + 1/2 cup lentils"),
        Calories = c(307, 269, 458, 211, 424),
        Protein_g = c(22, 7, 45, 9, 38),
        Carbs_g = c(16, 33, 56, 26, 57),
        Fat_g = c(18, 14, 7, 8, 13),
        Fiber_g = c(1, 7, 8, 3, 12)
      )
    ),
    
    Gastritis = list(
      foods_to_eat = data.frame(
        Food = c("Oatmeal", "Bananas", "Chicken breast", "White rice", "Yogurt",
                 "Papaya", "Zucchini", "Green beans", "Applesauce", "Lean turkey",
                 "Ginger tea", "Whole wheat bread"),
        Serving_Size = c("1 cup cooked", "1 medium", "4 oz", "1 cup cooked", "1 cup",
                         "1 cup", "1 cup", "1 cup", "1 cup", "4 oz",
                         "1 cup", "1 slice"),
        Calories = c(154, 105, 187, 205, 149,
                     62, 20, 44, 102, 153,
                     2, 69),
        Protein_g = c(6, 1, 35, 4, 8,
                      1, 1, 2, 0, 30,
                      0, 3),
        Carbs_g = c(27, 27, 0, 45, 11,
                    16, 4, 10, 28, 0,
                    0, 13),
        Fat_g = c(3, 0, 4, 0, 8,
                  0, 0, 0, 0, 3,
                  0, 1),
        Fiber_g = c(4, 3, 0, 1, 0,
                    3, 1, 4, 3, 0,
                    0, 2)
      ),
      foods_to_avoid = c("Spicy foods", "Citrus fruits", "Tomatoes", "Alcohol",
                         "Coffee", "Chocolate", "Fried foods", "High-fat foods", "Carbonated drinks"),
      sample_meal = data.frame(
        Meal = c("Breakfast", "Mid-Morning", "Lunch", "Evening", "Dinner"),
        Food = c("Oatmeal with banana", "Papaya and yogurt",
                 "Grilled chicken with white rice and zucchini", "Applesauce with whole wheat bread",
                 "Baked turkey with green beans and ginger tea"),
        Quantity = c("1 cup oatmeal + 1 banana", "1 cup papaya + 1 cup yogurt",
                     "4 oz chicken + 1 cup rice + 1 cup zucchini", "1 cup applesauce + 1 slice bread",
                     "4 oz turkey + 1 cup green beans + 1 cup tea"),
        Calories = c(259, 211, 412, 171, 199),
        Protein_g = c(7, 9, 40, 3, 32),
        Carbs_g = c(54, 27, 49, 41, 10),
        Fat_g = c(3, 8, 4, 1, 3),
        Fiber_g = c(7, 3, 2, 5, 4)
      )
    ),
    
    "Common Cold" = list(
      foods_to_eat = data.frame(
        Food = c("Chicken soup", "Oranges", "Ginger", "Garlic", "Honey",
                 "Green tea", "Spinach", "Berries", "Yogurt", "Salmon",
                 "Broccoli", "Sweet potato"),
        Serving_Size = c("1 cup", "1 medium", "1 tbsp fresh", "3 cloves", "1 tbsp",
                         "1 cup", "2 cups", "1 cup", "1 cup", "4 oz",
                         "1 cup", "1 medium"),
        Calories = c(86, 62, 5, 13, 64,2, 14, 84, 149, 206,55, 103),
        Protein_g = c(6, 1, 0, 1, 0, 0, 2, 1, 8, 23, 4, 2),
        Carbs_g = c(8, 15, 1, 3, 17, 0, 2, 21, 11, 0,11, 24),
        Fat_g = c(3, 0, 0, 0, 0, 0, 0, 0.5, 8, 13, 1, 0),
        Fiber_g = c(1, 3, 0, 0, 0, 0, 1, 4, 0, 0, 5, 4)
      ),
      foods_to_avoid = c("Alcohol", "Sugary snacks", "Fried foods", "Processed foods",
                         "High-sodium foods", "Dairy products (if congested)", "Red meat"),
      sample_meal = data.frame(
        Meal = c("Breakfast", "Mid-Morning", "Lunch", "Evening", "Dinner"),
        Food = c("Yogurt with berries and honey", "Orange with green tea",
                 "Chicken soup with spinach and garlic", "Sweet potato with broccoli",
                 "Grilled salmon with vegetables and ginger"),
        Quantity = c("1 cup yogurt + 1 cup berries + 1 tbsp honey", "1 orange + 1 cup tea",
                     "1 cup soup + 1 cup spinach + 2 cloves garlic", "1 potato + 1 cup broccoli",
                     "4 oz salmon + 2 cups vegetables + 1 tbsp ginger"),
        Calories = c(297, 64, 108, 158, 286),
        Protein_g = c(9, 1, 9, 6, 30),
        Carbs_g = c(49, 15, 13, 35, 19),
        Fat_g = c(8.5, 0, 3, 1, 14.5),
        Fiber_g = c(4, 3, 2, 9, 10)
      )
    ),
                     
                     Psoriasis = list(
                       foods_to_eat = data.frame(
                         Food = c("Salmon", "Sardines", "Turmeric", "Olive oil", "Leafy greens",
                                  "Berries", "Walnuts", "Flaxseeds", "Garlic", "Carrots",
                                  "Sweet potato", "Green tea"),
                         Serving_Size = c("4 oz", "3.75 oz", "1 tsp", "1 tbsp", "2 cups",
                                          "1 cup", "1 oz", "2 tbsp", "3 cloves", "1 cup",
                                          "1 medium", "1 cup"),
                         Calories = c(206, 208, 8, 119, 14, 84, 185, 75, 13, 52, 103, 2),
                         Protein_g = c(23, 25, 0, 0, 2, 1, 4, 3, 1, 1, 2, 0),
                         Carbs_g = c(0, 0, 2, 0, 2, 21, 4, 4, 3, 12, 24, 0),
                         Fat_g = c(13, 11, 0, 14, 0, 0.5, 18, 6, 0, 0, 0, 0),
                         Fiber_g = c(0, 0, 1, 0, 1, 4, 2, 4, 0, 4, 4, 0)
                       ),
                       foods_to_avoid = c("Red meat", "Dairy (if sensitive)", "Refined sugars",
                                          "Processed foods", "Alcohol", "Nightshades (if trigger)"),
                       sample_meal = data.frame(
                         Meal = c("Breakfast", "Mid-Morning", "Lunch", "Evening", "Dinner"),
                         Food = c("Berry smoothie with flaxseeds", "Green tea with walnuts",
                                  "Grilled salmon with vegetables", "Carrot sticks",
                                  "Baked fish with sweet potato"),
                         Quantity = c("2 cups + 2 tbsp seeds", "1 cup + 1 oz nuts",
                                      "4 oz + 2 cups vegetables", "1 cup",
                                      "4 oz + 1 potato + salad"),
                         Calories = c(280, 190, 450, 52, 480),
                         Protein_g = c(7, 4, 42, 1, 44),
                         Carbs_g = c(48, 4, 24, 12, 44),
                         Fat_g = c(10, 18, 22, 0, 18),
                         Fiber_g = c(12, 2, 8, 4, 10)
                       )
                     )
        )
        
        return(dietary_plans)
} 

# Disease Information Database
create_disease_info_database <- function() {
  disease_info <- list(
    Diabetes = list(
      full_name = "Diabetes Mellitus Type 2",
      description = "A chronic condition affecting how your body processes blood sugar (glucose). The body either doesn't produce enough insulin or can't effectively use the insulin it produces, leading to elevated blood sugar levels.",
      symptoms = c("Frequent urination", "Excessive thirst", "Unexplained weight loss", "Fatigue and weakness", 
                   "Blurred vision", "Slow-healing wounds", "Tingling in hands/feet", "Increased hunger"),
      causes = c("Insulin resistance", "Genetic factors", "Obesity and excess weight", "Sedentary lifestyle", 
                 "Age over 45", "Family history of diabetes", "High blood pressure", "Poor diet habits"),
      key_markers = c("Fasting glucose > 126 mg/dL", "HbA1c ≥ 6.5%", "Random glucose > 200 mg/dL", 
                      "Oral glucose tolerance test > 200 mg/dL"),
      treatment = c("Blood sugar monitoring regularly", "Oral medications (Metformin, Sulfonylureas)", 
                    "Insulin therapy if needed", "Regular exercise (150 min/week)", 
                    "Dietary management (low glycemic index foods)", "Weight loss if overweight", 
                    "Regular medical checkups", "Foot care and eye examinations")
    ),
    Hypertension = list(
      full_name = "High Blood Pressure (Hypertension)",
      description = "A condition where blood pressure is persistently elevated above normal levels, forcing the heart to work harder to pump blood.",
      symptoms = c("Headaches (especially in the morning)", "Dizziness and lightheadedness", "Blurred vision", 
                   "Chest pain", "Shortness of breath", "Nosebleeds", "Fatigue", "Irregular heartbeat"),
      causes = c("Age (risk increases with age)", "Genetics and family history", "Obesity and overweight", 
                 "High sodium diet", "Lack of physical exercise", "Chronic stress", "Smoking and tobacco use", 
                 "Excessive alcohol consumption", "Chronic kidney disease"),
      key_markers = c("Systolic BP ≥ 140 mmHg", "Diastolic BP ≥ 90 mmHg", "Consistent high readings over time"),
      treatment = c("ACE inhibitors medication", "Beta blockers", "Diuretics (water pills)", 
                    "Lifestyle modifications", "DASH diet (low sodium)", "Regular aerobic exercise", 
                    "Stress management techniques", "Weight loss", "Limit alcohol and quit smoking")
    ),
    Cancer = list(
      full_name = "Cancer (Various Types)",
      description = "Abnormal cell growth that can invade or spread to other parts of the body. Cancer occurs when cells divide uncontrollably and have the ability to infiltrate and destroy normal body tissue.",
      symptoms = c("Unexplained weight loss", "Persistent fatigue", "Pain in affected area", "Skin changes", 
                   "Unusual bleeding or discharge", "Persistent cough", "Changes in bowel/bladder habits", 
                   "Difficulty swallowing", "Lumps or thickening", "Night sweats"),
      causes = c("Genetic mutations", "Tobacco and smoking", "Alcohol consumption", "Obesity", 
                 "UV radiation exposure", "Certain viral infections (HPV, Hepatitis)", 
                 "Environmental toxins and chemicals", "Age (risk increases)", "Chronic inflammation"),
      key_markers = c("Tumor markers (CA 19-9, CEA, PSA)", "Abnormal blood counts", "Imaging abnormalities", 
                      "Biopsy showing malignant cells", "Elevated LDH levels"),
      treatment = c("Surgical removal of tumor", "Chemotherapy", "Radiation therapy", "Immunotherapy", 
                    "Targeted therapy", "Hormone therapy", "Stem cell transplant", "Palliative care")
    ),
    "Liver Damage" = list(
      full_name = "Liver Disease / Hepatic Dysfunction",
      description = "Damage to liver tissue affecting its function in detoxification, protein synthesis, and metabolism. The liver may become inflamed, scarred, or develop fatty deposits.",
      symptoms = c("Jaundice (yellow skin and eyes)", "Abdominal pain and swelling", "Swelling in legs and ankles", 
                   "Itchy skin", "Dark urine", "Pale stool", "Chronic fatigue", "Nausea and vomiting", 
                   "Loss of appetite", "Easy bruising"),
      causes = c("Alcohol abuse", "Viral hepatitis (A, B, C)", "Fatty liver disease (NAFLD)", 
                 "Certain medications and toxins", "Autoimmune conditions", "Genetic disorders (Hemochromatosis)", 
                 "Obesity", "Diabetes"),
      key_markers = c("Elevated ALT/AST (>40 U/L)", "Elevated bilirubin (>1.2 mg/dL)", "Low albumin", 
                      "Prolonged PT/INR", "Elevated ALP", "Jaundice visible"),
      treatment = c("Abstain from alcohol completely", "Antiviral medications (for hepatitis)", 
                    "Immunosuppressants (if autoimmune)", "Dietary modifications (low fat)", 
                    "Weight management", "Avoid hepatotoxic drugs", "Liver transplant (if severe)", 
                    "Regular monitoring of liver function")
    ),
    "Brain Stroke" = list(
      full_name = "Cerebrovascular Accident (Stroke)",
      description = "Interruption of blood supply to the brain causing brain cells to die. Can be ischemic (blocked artery) or hemorrhagic (burst blood vessel).",
      symptoms = c("Sudden numbness (face, arm, leg)", "Confusion and difficulty speaking", 
                   "Vision problems in one or both eyes", "Difficulty walking and loss of balance", 
                   "Severe headache", "Dizziness", "Loss of coordination", "Facial drooping"),
      causes = c("Blocked artery (ischemic stroke)", "Burst blood vessel (hemorrhagic)", "High blood pressure", 
                 "Diabetes", "Heart disease and atrial fibrillation", "Smoking", "High cholesterol", 
                 "Obesity", "Excessive alcohol", "Family history"),
      key_markers = c("Sudden onset symptoms", "CT/MRI showing brain infarct or hemorrhage", 
                      "Neurological deficits", "FAST test positive (Face, Arms, Speech, Time)"),
      treatment = c("Emergency tPA (clot-buster) within 4.5 hours", "Mechanical thrombectomy", 
                    "Blood pressure control", "Anticoagulants and antiplatelets", 
                    "Physical rehabilitation therapy", "Speech therapy", "Occupational therapy", 
                    "Surgery (if hemorrhagic)", "Preventive medications")
    ),
    "Heart Attack" = list(
      full_name = "Myocardial Infarction",
      description = "Blockage of blood flow to the heart muscle causing tissue damage. Occurs when a coronary artery becomes blocked, usually by a blood clot.",
      symptoms = c("Severe chest pain or pressure", "Pain in arms, neck, jaw, back", "Shortness of breath", 
                   "Cold sweats", "Nausea and vomiting", "Lightheadedness", "Fatigue", "Anxiety"),
      causes = c("Coronary artery disease", "Plaque buildup in arteries", "Blood clot formation", 
                 "Smoking", "High cholesterol", "Hypertension", "Diabetes", "Obesity", 
                 "Stress", "Sedentary lifestyle", "Age and family history"),
      key_markers = c("Elevated troponin (>0.04 ng/mL)", "ECG showing ST elevation", "Chest pain lasting >20 minutes", 
                      "Elevated CK-MB"),
      treatment = c("Aspirin immediately", "Angioplasty and stenting", "Coronary artery bypass surgery", 
                    "Beta blockers", "ACE inhibitors", "Statins for cholesterol", "Anticoagulants", 
                    "Cardiac rehabilitation program", "Lifestyle modifications")
    ),
    Tuberculosis = list(
      full_name = "Tuberculosis (TB)",
      description = "Bacterial infection primarily affecting the lungs, caused by Mycobacterium tuberculosis. Can spread through the air when infected person coughs or sneezes.",
      symptoms = c("Persistent cough (>3 weeks)", "Coughing up blood", "Chest pain", "Unintentional weight loss", 
                   "Fatigue and weakness", "Fever", "Night sweats", "Chills", "Loss of appetite"),
      causes = c("Mycobacterium tuberculosis bacteria", "Airborne transmission", "Weakened immune system", 
                 "Close contact with infected person", "HIV/AIDS", "Malnutrition", "Diabetes", 
                 "Living in crowded conditions", "Travel to high-prevalence areas"),
      key_markers = c("Positive TB skin test (Mantoux)", "Chest X-ray showing lung infiltrates", 
                      "Sputum culture positive for TB", "Persistent cough >3 weeks", "Night sweats", 
                      "Weight loss", "GeneXpert MTB/RIF test positive"),
      treatment = c("Isoniazid (6-9 months)", "Rifampin", "Ethambutol", "Pyrazinamide", 
                    "6-month intensive treatment", "Directly observed therapy (DOT)", 
                    "Isolation during infectious period", "Nutritional support", "Regular monitoring")
    ),
    Cataracts = list(
      full_name = "Cataracts",
      description = "Clouding of the eye's natural lens causing vision impairment. Most commonly occurs with aging but can also result from injury or other conditions.",
      symptoms = c("Cloudy or blurry vision", "Glare and halos around lights", "Faded colors", 
                   "Poor night vision", "Double vision in one eye", "Frequent prescription changes", 
                   "Difficulty reading"),
      causes = c("Aging (most common)", "Diabetes", "Excessive UV exposure", "Smoking", "Eye trauma", 
                 "Prolonged corticosteroid use", "Genetic factors", "Previous eye surgery", "Radiation exposure"),
      key_markers = c("Cloudy or opaque lens on examination", "Decreased visual acuity", "Glare sensitivity", 
                      "Slit-lamp examination showing lens opacity"),
      treatment = c("Corrective eyeglasses (early stages)", "Magnifying lenses", "Brighter lighting", 
                    "Cataract surgery (phacoemulsification)", "Intraocular lens (IOL) implant", 
                    "Post-operative care", "UV protection", "Regular eye exams")
    ),
    Eczema = list(
      full_name = "Atopic Dermatitis (Eczema)",
      description = "Chronic inflammatory skin condition causing itchy, red, and inflamed patches. Often begins in childhood but can occur at any age.",
      symptoms = c("Itchy red patches", "Dry, scaly skin", "Skin inflammation", "Cracked skin", 
                   "Thickened skin areas", "Small raised bumps", "Oozing and crusting", "Sensitive skin"),
      causes = c("Genetic factors", "Immune system dysfunction", "Environmental triggers (pollen, pets)", 
                 "Food allergens", "Stress", "Dry skin", "Irritants (soaps, detergents)", 
                 "Temperature changes", "Hormonal changes"),
      key_markers = c("Itchy red patches on skin", "Dry scaly areas", "Skin inflammation visible", 
                      "Elevated IgE levels", "Positive allergy tests"),
      treatment = c("Topical corticosteroids", "Moisturizers (emollients)", "Antihistamines for itching", 
                    "Immunomodulators (tacrolimus)", "Avoid triggers", "Gentle fragrance-free products", 
                    "Wet wrap therapy", "Phototherapy", "Systemic medications if severe")
    ),
    Psoriasis = list(
      full_name = "Psoriasis",
      description = "Autoimmune condition causing rapid skin cell buildup, forming thick, silvery scales and itchy, dry, red patches that are sometimes painful.",
      symptoms = c("Red patches with silvery scales", "Dry cracked skin that may bleed", "Itching and burning", 
                   "Thickened nails", "Swollen and stiff joints", "Plaques on scalp", "Soreness"),
      causes = c("Immune system overactivity", "Genetic predisposition", "Stress", "Skin injuries (cuts, scrapes)", 
                 "Infections (strep throat)", "Smoking", "Heavy alcohol consumption", "Certain medications", 
                 "Cold weather"),
      key_markers = c("Red patches with silvery scales", "Dry cracked skin", "Nail changes (pitting, thickening)", 
                      "Joint pain (psoriatic arthritis)", "Skin biopsy showing characteristic changes"),
      treatment = c("Topical treatments (corticosteroids)", "Vitamin D analogues", "Phototherapy (UV light)", 
                    "Systemic medications (methotrexate)", "Biologic drugs (TNF inhibitors)", 
                    "Stress management", "Moisturizers", "Avoid triggers", "Coal tar preparations")
    ),
    "Common Cold" = list(
      full_name = "Common Cold (Viral Upper Respiratory Infection)",
      description = "A viral infection of the upper respiratory tract, primarily affecting the nose and throat. Usually harmless and resolves within 7-10 days.",
      symptoms = c("Runny or stuffy nose", "Sore throat", "Cough", "Sneezing", "Mild headache", 
                   "Body aches", "Low-grade fever", "Fatigue"),
      causes = c("Rhinovirus (most common)", "Coronavirus", "Airborne droplets", "Direct contact with infected person", 
                 "Touching contaminated surfaces", "Weakened immune system", "Stress", "Lack of sleep"),
      key_markers = c("Nasal congestion", "Clear nasal discharge", "Sore throat", "Normal or low-grade fever"),
      treatment = c("Rest and adequate sleep", "Stay hydrated", "Warm liquids (soup, tea)", "Over-the-counter pain relievers", 
                    "Decongestants", "Honey for cough", "Gargle with salt water", "Vitamin C supplements")
    ),
    Influenza = list(
      full_name = "Influenza (Flu)",
      description = "A contagious respiratory illness caused by influenza viruses, more severe than common cold with sudden onset of symptoms.",
      symptoms = c("High fever (100-104°F)", "Severe body aches", "Headache", "Dry cough", "Sore throat", 
                   "Extreme fatigue", "Chills", "Nasal congestion"),
      causes = c("Influenza A virus", "Influenza B virus", "Airborne transmission", "Close contact with infected people", 
                 "Seasonal changes", "Weakened immune system"),
      key_markers = c("Sudden high fever", "Severe muscle aches", "Rapid antigen test positive", "PCR test positive"),
      treatment = c("Antiviral medications (Tamiflu, Relenza)", "Rest and isolation", "Fluids and hydration", 
                    "Fever reducers", "Pain relievers", "Annual flu vaccine (prevention)", "Symptomatic treatment")
    ),
    Gastritis = list(
      full_name = "Gastritis (Stomach Inflammation)",
      description = "Inflammation of the stomach lining causing pain and digestive issues. Can be acute or chronic.",
      symptoms = c("Upper abdominal pain", "Nausea and vomiting", "Loss of appetite", "Bloating", 
                   "Indigestion", "Black tarry stools", "Feeling full quickly", "Hiccups"),
      causes = c("H. pylori bacteria", "Excessive alcohol", "NSAID use (aspirin, ibuprofen)", "Stress", 
                 "Autoimmune disorders", "Bile reflux", "Viral infections"),
      key_markers = c("Abdominal pain and tenderness", "Endoscopy showing inflammation", "H. pylori test positive", 
                      "Elevated gastric acid"),
      treatment = c("Proton pump inhibitors (PPIs)", "H2 blockers", "Antacids", "Antibiotics (for H. pylori)", 
                    "Avoid NSAIDs", "Dietary changes", "Avoid spicy/acidic foods", "Stress management")
    ),
    Migraine = list(
      full_name = "Migraine Headache",
      description = "A neurological condition characterized by intense, debilitating headaches often accompanied by nausea, vomiting, and sensitivity to light and sound.",
      symptoms = c("Severe throbbing headache", "Nausea and vomiting", "Light sensitivity", "Sound sensitivity", 
                   "Visual disturbances (aura)", "Dizziness", "Fatigue", "Neck stiffness"),
      causes = c("Genetic factors", "Hormonal changes", "Stress and anxiety", "Certain foods (chocolate, cheese)", 
                 "Caffeine", "Sleep disturbances", "Weather changes", "Strong odors", "Bright lights"),
      key_markers = c("Unilateral throbbing pain", "Aura (visual disturbances)", "Nausea", "Photophobia", "Phonophobia"),
      treatment = c("Triptans (sumatriptan)", "Pain relievers (NSAIDs)", "Anti-nausea medications", "Preventive medications", 
                    "Avoid triggers", "Regular sleep schedule", "Stress management", "Biofeedback therapy")
    ),
    Asthma = list(
      full_name = "Asthma (Chronic Respiratory Disease)",
      description = "A chronic condition where airways narrow and swell, producing extra mucus, making breathing difficult.",
      symptoms = c("Shortness of breath", "Wheezing", "Chest tightness", "Coughing (especially at night)", 
                   "Difficulty breathing", "Rapid breathing", "Fatigue"),
      causes = c("Genetic predisposition", "Allergens (pollen, dust mites)", "Air pollution", "Respiratory infections", 
                 "Exercise", "Cold air", "Stress", "Smoking", "Occupational irritants"),
      key_markers = c("Wheezing on examination", "Reduced peak flow", "Spirometry showing obstruction", 
                      "Positive methacholine challenge"),
      treatment = c("Quick-relief inhalers (albuterol)", "Long-term control medications", "Inhaled corticosteroids", 
                    "Avoid triggers", "Allergy medications", "Breathing exercises", "Action plan", "Regular monitoring")
    ),
    Arthritis = list(
      full_name = "Arthritis (Joint Inflammation)",
      description = "Inflammation of one or more joints causing pain and stiffness that typically worsens with age.",
      symptoms = c("Joint pain", "Stiffness (especially morning)", "Swelling", "Redness", "Decreased range of motion", 
                   "Warmth in joints", "Fatigue", "Difficulty moving"),
      causes = c("Age-related wear and tear", "Autoimmune disorders", "Genetic factors", "Previous joint injury", 
                 "Obesity", "Infections", "Repetitive stress"),
      key_markers = c("Joint tenderness", "Swelling and inflammation", "X-ray showing joint damage", 
                      "Elevated ESR/CRP", "Rheumatoid factor positive"),
      treatment = c("NSAIDs", "Corticosteroids", "Disease-modifying drugs (DMARDs)", "Physical therapy", 
                    "Exercise", "Weight management", "Hot/cold therapy", "Joint replacement surgery (severe cases)")
    ),
    "Cardiac Arrest" = list(
      full_name = "Cardiac Arrest",
      description = "Sudden loss of heart function, breathing, and consciousness due to electrical malfunction in the heart.",
      symptoms = c("Sudden collapse", "No pulse", "No breathing", "Loss of consciousness", "No response"),
      causes = c("Heart attack", "Electrical problems in heart", "Cardiomyopathy", "Heart valve disease", 
                 "Congenital heart disease", "Drug overdose", "Severe trauma", "Electrolyte imbalances"),
      key_markers = c("No pulse", "No breathing", "Unconsciousness", "ECG showing ventricular fibrillation"),
      treatment = c("Immediate CPR", "Defibrillation (AED)", "Emergency medical services", "Advanced cardiac life support", 
                    "Post-resuscitation care", "Implantable cardioverter-defibrillator (ICD)", "Treating underlying cause")
    ),
    Paralysis = list(
      full_name = "Paralysis (Loss of Motor Function)",
      description = "Loss of muscle function in part of the body, can be partial or complete, temporary or permanent.",
      symptoms = c("Loss of movement", "Numbness", "Tingling sensation", "Loss of sensation", "Muscle weakness", 
                   "Difficulty speaking", "Loss of bowel/bladder control"),
      causes = c("Stroke", "Spinal cord injury", "Multiple sclerosis", "Cerebral palsy", "Brain injury", 
                 "Guillain-Barré syndrome", "Polio", "Nerve damage"),
      key_markers = c("Loss of motor function", "Abnormal reflexes", "MRI showing nerve damage", 
                      "EMG showing muscle problems"),
      treatment = c("Physical therapy", "Occupational therapy", "Mobility aids", "Medications for spasticity", 
                    "Surgery (in some cases)", "Nerve stimulation", "Assistive devices", "Rehabilitation programs")
    ),
    Anemia = list(
      full_name = "Anemia (Low Red Blood Cells)",
      description = "Condition where blood lacks adequate healthy red blood cells to carry oxygen to body's tissues.",
      symptoms = c("Fatigue and weakness", "Pale skin", "Shortness of breath", "Dizziness", "Cold hands and feet", 
                   "Headaches", "Irregular heartbeat", "Chest pain"),
      causes = c("Iron deficiency", "Vitamin B12 deficiency", "Folate deficiency", "Chronic diseases", 
                 "Blood loss", "Inherited disorders", "Bone marrow problems", "Hemolysis"),
      key_markers = c("Low hemoglobin (<12 g/dL women, <13 g/dL men)", "Low hematocrit", "Low RBC count", 
                      "Low ferritin", "Pale conjunctiva"),
      treatment = c("Iron supplements", "Vitamin B12 injections", "Folic acid supplements", "Dietary changes", 
                    "Treat underlying cause", "Blood transfusion (severe cases)", "Erythropoietin injections", 
                    "Bone marrow transplant (rare cases)")
    ),
    Glaucoma = list(
      full_name = "Glaucoma",
      description = "Group of eye conditions damaging the optic nerve, often due to high eye pressure, leading to vision loss.",
      symptoms = c("Gradual vision loss", "Eye pain", "Blurred vision", "Halos around lights", "Eye redness", 
                   "Nausea and vomiting", "Tunnel vision", "Headaches"),
      causes = c("High intraocular pressure", "Age over 60", "Family history", "Diabetes", "Eye injury", 
                 "Prolonged corticosteroid use", "Thin corneas", "Extreme nearsightedness"),
      key_markers = c("Elevated intraocular pressure (>21 mmHg)", "Optic nerve damage", "Visual field loss", 
                      "Increased cup-to-disc ratio"),
      treatment = c("Eye drops to reduce pressure", "Oral medications", "Laser therapy", "Surgery (trabeculectomy)", 
                    "Drainage implants", "Regular monitoring", "Lifestyle modifications")
    ),
    "Macular Degeneration" = list(
      full_name = "Age-Related Macular Degeneration (AMD)",
      description = "Eye disease causing loss of central vision due to damage to the macula, affecting ability to see fine details.",
      symptoms = c("Blurred central vision", "Dark or empty areas in center of vision", "Distorted vision", 
                   "Difficulty recognizing faces", "Need for brighter light", "Difficulty adapting to low light"),
      causes = c("Age (over 50)", "Genetics", "Smoking", "Obesity", "High blood pressure", "High cholesterol", 
                 "UV light exposure", "Race (more common in Caucasians)"),
      key_markers = c("Drusen (yellow deposits)", "Vision distortion on Amsler grid", "OCT showing retinal changes", 
                      "Central vision loss"),
      treatment = c("Anti-VEGF injections", "Laser therapy", "Photodynamic therapy", "Vitamins and minerals (AREDS)", 
                    "Low vision aids", "Quit smoking", "Healthy diet", "Regular eye exams")
    ),
    Conjunctivitis = list(
      full_name = "Conjunctivitis (Pink Eye)",
      description = "Inflammation of the conjunctiva (clear tissue covering white part of eye), causing redness and irritation.",
      symptoms = c("Eye redness", "Itchy eyes", "Gritty feeling", "Discharge (crusty)", "Tearing", 
                   "Swollen eyelids", "Light sensitivity", "Blurred vision"),
      causes = c("Viral infection", "Bacterial infection", "Allergies", "Chemical irritants", "Foreign object", 
                 "Blocked tear duct (infants)"),
      key_markers = c("Red/pink eye", "Discharge", "Swollen conjunctiva", "Culture positive (bacterial)"),
      treatment = c("Antibiotic eye drops (bacterial)", "Antiviral medications (viral)", "Antihistamine drops (allergic)", 
                    "Warm compresses", "Artificial tears", "Avoid contact lenses", "Good hygiene", "Self-limiting (viral)")
    ),
    Melanoma = list(
      full_name = "Melanoma (Skin Cancer)",
      description = "Most serious type of skin cancer developing in melanocytes (pigment-producing cells).",
      symptoms = c("New mole or growth", "Change in existing mole", "Irregular borders", "Multiple colors", 
                   "Asymmetrical shape", "Diameter >6mm", "Evolving size/shape/color", "Itching or bleeding"),
      causes = c("UV radiation exposure", "Sunburns", "Tanning beds", "Fair skin", "Many moles", 
                 "Family history", "Weakened immune system", "Age"),
      key_markers = c("ABCDE criteria (Asymmetry, Border, Color, Diameter, Evolution)", "Biopsy showing melanoma cells", 
                      "Irregular pigmented lesion"),
      treatment = c("Surgical excision", "Lymph node removal", "Immunotherapy", "Targeted therapy", "Chemotherapy", 
                    "Radiation therapy", "Clinical trials", "Regular skin checks", "Sun protection")
    )
  )
  
  return(disease_info)
}

# Function to extract text from PDF
extract_text_from_pdf <- function(file_path) {
  tryCatch({
    text <- pdf_text(file_path)
    text <- paste(text, collapse = " ")
    return(text)
  }, error = function(e) {
    return(paste("Error reading PDF:", e$message))
  })
}

# Function to extract text from image using OCR
extract_text_from_image <- function(file_path) {
  tryCatch({
    image <- image_read(file_path)
    text <- image_ocr(image)
    return(text)
  }, error = function(e) {
    return(paste("Error reading image:", e$message))
  })
}

# Function to extract numeric values from text
extract_numeric_value <- function(text, pattern) {
  matches <- str_extract_all(text, paste0(pattern, "\\s*[:=]?\\s*([0-9]+\\.?[0-9]*)"))[[1]]
  if (length(matches) > 0) {
    value <- as.numeric(str_extract(matches[1], "[0-9]+\\.?[0-9]*"))
    return(value)
  }
  return(NA)
}

# Function to analyze medical report with actual value extraction
analyze_medical_report <- function(report_text, disease_info_db) {
  report_lower <- tolower(report_text)
  
  findings <- list()
  detected_values <- list()
  
  # Extract actual test values
  glucose_val <- extract_numeric_value(report_lower, "glucose|blood sugar|fasting glucose")
  hba1c_val <- extract_numeric_value(report_lower, "hba1c|hemoglobin a1c|glycated hemoglobin")
  
  alt_val <- extract_numeric_value(report_lower, "alt|sgpt|alanine")
  ast_val <- extract_numeric_value(report_lower, "ast|sgot|aspartate")
  bilirubin_val <- extract_numeric_value(report_lower, "bilirubin|total bilirubin")
  
  troponin_val <- extract_numeric_value(report_lower, "troponin")
  cholesterol_val <- extract_numeric_value(report_lower, "cholesterol|total cholesterol")
  ldl_val <- extract_numeric_value(report_lower, "ldl")
  
  hemoglobin_val <- extract_numeric_value(report_lower, "hemoglobin|hb|hgb")
  
  systolic_bp <- extract_numeric_value(report_lower, "systolic|bp|blood pressure")
  
  # Analyze Diabetes
  if (!is.na(glucose_val) || !is.na(hba1c_val)) {
    diabetes_risk <- FALSE
    details <- c()
    
    if (!is.na(glucose_val)) {
      detected_values$"Fasting Glucose" <- paste(glucose_val, "mg/dL")
      if (glucose_val >= 126) {
        diabetes_risk <- TRUE
        details <- c(details, paste("Elevated fasting glucose:", glucose_val, "mg/dL (Normal: 70-99)"))
      } else if (glucose_val >= 100) {
        details <- c(details, paste("Pre-diabetic glucose:", glucose_val, "mg/dL (Normal: 70-99)"))
      }
    }
    
    if (!is.na(hba1c_val)) {
      detected_values$"HbA1c" <- paste(hba1c_val, "%")
      if (hba1c_val >= 6.5) {
        diabetes_risk <- TRUE
        details <- c(details, paste("Elevated HbA1c:", hba1c_val, "% (Normal: <5.7%)"))
      } else if (hba1c_val >= 5.7) {
        details <- c(details, paste("Pre-diabetic HbA1c:", hba1c_val, "% (Normal: <5.7%)"))
      }
    }
    
    if (diabetes_risk) {
      findings$Diabetes <- paste(details, collapse = ". ")
    }
  }
  
  # Analyze Liver Function
  if (!is.na(alt_val) || !is.na(ast_val) || !is.na(bilirubin_val)) {
    liver_risk <- FALSE
    details <- c()
    
    if (!is.na(alt_val)) {
      detected_values$"ALT" <- paste(alt_val, "U/L")
      if (alt_val > 56) {
        liver_risk <- TRUE
        details <- c(details, paste("Elevated ALT:", alt_val, "U/L (Normal: 7-56)"))
      }
    }
    
    if (!is.na(ast_val)) {
      detected_values$"AST" <- paste(ast_val, "U/L")
      if (ast_val > 40) {
        liver_risk <- TRUE
        details <- c(details, paste("Elevated AST:", ast_val, "U/L (Normal: 10-40)"))
      }
    }
    
    if (!is.na(bilirubin_val)) {
      detected_values$"Bilirubin" <- paste(bilirubin_val, "mg/dL")
      if (bilirubin_val > 1.2) {
        liver_risk <- TRUE
        details <- c(details, paste("Elevated Bilirubin:", bilirubin_val, "mg/dL (Normal: 0.1-1.2)"))
      }
    }
    
    if (liver_risk) {
      findings$"Liver Damage" <- paste(details, collapse = ". ")
    }
  }
  
  # Analyze Cardiac Risk
  if (!is.na(troponin_val) || !is.na(cholesterol_val) || !is.na(ldl_val)) {
    cardiac_risk <- FALSE
    details <- c()
    
    if (!is.na(troponin_val)) {
      detected_values$"Troponin" <- paste(troponin_val, "ng/mL")
      if (troponin_val > 0.04) {
        cardiac_risk <- TRUE
        details <- c(details, paste("Elevated Troponin:", troponin_val, "ng/mL (Normal: <0.04) - Possible heart damage"))
      }
    }
    
    if (!is.na(cholesterol_val)) {
      detected_values$"Total Cholesterol" <- paste(cholesterol_val, "mg/dL")
      if (cholesterol_val > 200) {
        cardiac_risk <- TRUE
        details <- c(details, paste("High Cholesterol:", cholesterol_val, "mg/dL (Normal: <200)"))
      }
    }
    
    if (!is.na(ldl_val)) {
      detected_values$"LDL Cholesterol" <- paste(ldl_val, "mg/dL")
      if (ldl_val > 130) {
        cardiac_risk <- TRUE
        details <- c(details, paste("High LDL:", ldl_val, "mg/dL (Normal: <100)"))
      }
    }
    
    if (cardiac_risk) {
      findings$"Heart Attack" <- paste(details, collapse = ". ")
    }
  }
  
  # Analyze Anemia
  if (!is.na(hemoglobin_val)) {
    detected_values$"Hemoglobin" <- paste(hemoglobin_val, "g/dL")
    if (hemoglobin_val < 12) {
      findings$Anemia <- paste("Low Hemoglobin:", hemoglobin_val, "g/dL (Normal: Men 13.5-17.5, Women 12-15.5)")
    }
  }
  
  # Analyze Blood Pressure
  if (!is.na(systolic_bp)) {
    detected_values$"Blood Pressure" <- paste(systolic_bp, "mmHg (systolic)")
    if (systolic_bp >= 140) {
      findings$Hypertension <- paste("Elevated Blood Pressure:", systolic_bp, "mmHg systolic (Normal: <120)")
    }
  }
  
  # Check for disease keywords in report
  disease_keywords <- list(
    Tuberculosis = c("tuberculosis", "tb", "mycobacterium", "chest x-ray.*infiltrate"),
    Cancer = c("cancer", "tumor", "malignant", "carcinoma", "metastasis", "biopsy.*positive"),
    "Brain Stroke" = c("stroke", "cerebrovascular accident", "cva", "brain infarct"),
    Cataracts = c("cataract", "lens opacity", "cloudy lens"),
    Eczema = c("eczema", "atopic dermatitis", "skin inflammation"),
    Psoriasis = c("psoriasis", "plaque psoriasis", "scaly patches")
  )
  
  for (disease in names(disease_keywords)) {
    if (any(sapply(disease_keywords[[disease]], function(x) grepl(x, report_lower)))) {
      if (is.null(findings[[disease]])) {
        findings[[disease]] <- paste(disease, "mentioned in report - Clinical diagnosis present")
      }
    }
  }
  
  # If no specific findings
  if (length(findings) == 0) {
    findings$"General Health Assessment" <- "No abnormal values detected in the report. All extracted values appear within normal ranges. Please consult your healthcare provider for detailed interpretation."
  }
  
  return(list(findings = findings, detected_values = detected_values))
}

# ML Model Training
train_disease_model <- function(data) {
  features <- data[, -ncol(data)]
  target <- as.factor(data$Disease)
  set.seed(123)
  model <- randomForest(x = features, y = target, ntree = 150, importance = TRUE)
  return(model)
}

predict_disease <- function(model, symptoms_vector) {
  prediction <- predict(model, symptoms_vector, type = "prob")
  predicted_disease <- names(which.max(prediction))
  confidence <- max(prediction) * 100
  return(list(disease = predicted_disease, confidence = confidence, probabilities = prediction))
}

calculate_total_nutrition <- function(meal_plan) {
  total <- data.frame(
    Total_Calories = sum(meal_plan$Calories),
    Total_Protein = sum(meal_plan$Protein_g),
    Total_Carbs = sum(meal_plan$Carbs_g),
    Total_Fat = sum(meal_plan$Fat_g),
    Total_Fiber = sum(meal_plan$Fiber_g)
  )
  return(total)
}

# Initialize data
disease_data <- create_disease_symptom_data()
dietary_data <- create_dietary_data()
disease_info_db <- create_disease_info_database()
ml_model <- train_disease_model(disease_data)

# UI
ui <- fluidPage(
  tags$head(
    tags$style(HTML("
      .btn-primary { background-color: #667eea; border-color: #667eea; }
      .btn-primary:hover { background-color: #5568d3; }
      .btn-success { background-color: #28a745; }
      .btn-info { background-color: #17a2b8; }
      h3 { color: #667eea; margin-top: 20px; }
      .result-box { 
        background-color: #f8f9fa; 
        padding: 20px; 
        border-radius: 10px; 
        border-left: 4px solid #667eea;
        margin-bottom: 20px;
      }
      .info-card {
        background: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 15px;
      }
      .marker-badge {
        display: inline-block;
        padding: 5px 10px;
        background: #e7f3ff;
        border-radius: 5px;
        margin: 5px;
        font-size: 13px;
      }
    "))
  ),
  
  titlePanel(
    div(style = "background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                 padding: 30px; color: white; border-radius: 10px;",
        h1("🏥 Advanced AI Medical Diagnosis System"),
        h4("Comprehensive Disease Analysis with Diagnostic Report Upload")
    )
  ),
  
  tabsetPanel(
    # Tab 1: Symptom-Based Diagnosis
    tabPanel("🔍 Symptom Analysis",
             br(),
             sidebarLayout(
               sidebarPanel(
                 h3("Select Your Symptoms"),
                 hr(),
                 h5("General Symptoms:"),
                 checkboxInput("fatigue", "Fatigue"),
                 checkboxInput("tiredness", "Tiredness/Weakness"),
                 checkboxInput("sleeplessness", "Sleeplessness"),
                 checkboxInput("fever", "Fever"),
                 checkboxInput("sweating", "Excessive Sweating"),
                 
                 hr(),
                 h5("Neurological:"),
                 checkboxInput("headache", "Headache"),
                 checkboxInput("dizziness", "Dizziness"),
                 checkboxInput("blurred_vision", "Blurred Vision"),
                 checkboxInput("numbness", "Numbness"),
                 checkboxInput("confusion", "Confusion"),
                 
                 hr(),
                 h5("Respiratory & Cardiac:"),
                 checkboxInput("cough", "Cough"),
                 checkboxInput("shortness_breath", "Shortness of Breath"),
                 checkboxInput("chest_pain", "Chest Pain"),
                 checkboxInput("irregular_heartbeat", "Irregular Heartbeat"),
                 
                 hr(),
                 h5("Digestive:"),
                 checkboxInput("nausea", "Nausea"),
                 checkboxInput("abdominal_pain", "Abdominal Pain"),
                 checkboxInput("loss_appetite", "Loss of Appetite"),
                 checkboxInput("rapid_hunger", "Rapid Hunger"),
                 checkboxInput("jaundice", "Jaundice"),
                 
                 hr(),
                 h5("Eyes & Skin:"),
                 checkboxInput("eye_pain", "Eye Pain"),
                 checkboxInput("eye_redness", "Eye Redness"),
                 checkboxInput("light_sensitivity", "Light Sensitivity"),
                 checkboxInput("skin_rash", "Skin Rash"),
                 checkboxInput("itching", "Itching"),
                 checkboxInput("skin_lesions", "Skin Lesions"),
                 
                 hr(),
                 h5("Other:"),
                 checkboxInput("joint_pain", "Joint Pain"),
                 checkboxInput("freq_urination", "Frequent Urination"),
                 checkboxInput("thirst", "Increased Thirst"),
                 checkboxInput("weight_loss", "Weight Loss"),
                 checkboxInput("pale_skin", "Pale Skin"),
                 checkboxInput("swelling", "Swelling"),
                 
                 br(),
                 actionButton("diagnose", "🔬 Diagnose Disease", 
                              class = "btn-primary btn-lg btn-block")
               ),
               
               mainPanel(uiOutput("diagnosis_output"))
             )
    ),
    
    # Tab 2: Report Upload & Analysis
    tabPanel("📄 Diagnostic Report Analysis",
             br(),
             fluidRow(
               column(12,
                      div(class = "result-box",
                          h3("Upload Your Diagnostic Report"),
                          p("Upload your lab report, test results, or medical documents in PDF or Image format up to 5.0 MB size."),
                          fileInput("report_file", "Choose File (PDF or Image)",
                                    accept = c(".pdf", ".jpg", ".jpeg", ".png", ".tiff")),
                          actionButton("analyze_report", "🔍 Analyze Report", 
                                       class = "btn-info btn-lg")
                      )
               )
             ),
             fluidRow(
               column(12,
                      uiOutput("report_analysis_output")
               )
             )
    ),
    
    # Tab 3: Dietary Plan
    tabPanel("🥗 Dietary Plans",
             br(),
             sidebarLayout(
               sidebarPanel(
                 h3("Select Disease"),
                 selectInput("disease_select", "Choose Disease:",
                             choices = c("Diabetes", "Hypertension", "Cancer", 
                                         "Liver Damage", "Cataracts", "Eczema", "Psoriasis",
                                         "Common Cold", "Influenza", "Gastritis", "Migraine",
                                         "Asthma", "Arthritis", "Brain Stroke", "Cardiac Arrest",
                                         "Heart Attack", "Tuberculosis", "Paralysis", "Anemia",
                                         "Glaucoma", "Macular Degeneration", "Conjunctivitis", "Melanoma"),
                             selected = "Diabetes"),
                 br(),
                 actionButton("get_diet", "🍽️ Get Dietary Plan", 
                              class = "btn-success btn-lg btn-block")
               ),
               
               mainPanel(uiOutput("diet_output"))
             )
    ),
    
    # Tab 4: About


tabPanel("ℹ️ About",
         br(),
         div(style = "padding: 30px;",
             h2("About This Advanced Medical AI System"),
             br(),
             
             div(class = "result-box",
                 h3("🎯 System Features"),
                 tags$ul(
                   tags$li(strong("Comprehensive Disease Coverage:"), " 23 major diseases including 
                              chronic conditions, cardiovascular diseases, cancers, and infectious diseases"),
                   tags$li(strong("31 Clinical Symptoms:"), " Extended symptom database for accurate diagnosis"),
                   tags$li(strong("Random Forest ML Algorithm:"), " 150-tree ensemble for high accuracy predictions"),
                   tags$li(strong("Personalized Dietary Plans:"), " Disease-specific nutrition recommendations"),
                   tags$li(strong("Complete Nutritional Analysis:"), " Tracks calories, macronutrients, and fiber"),
                   tags$li(strong("Evidence-Based Recommendations:"), " All plans based on medical research")
                 )
             ),
             
             div(class = "result-box",
                 h3("🏥 Diseases Covered For Dietary Plans"),
                 div(style = "display: grid; grid-template-columns: 1fr 1fr; gap: 10px;",
                     tags$ul(
                       tags$li("Diabetes Type 2"),
                       tags$li("Hypertension"),
                       tags$li("Common Cold"),
                       tags$li("Gastritis"),
                       tags$li("Cancer (General)"),
                       tags$li("Brain Stroke"),
                       tags$li("Heart Attack"),
                       tags$li("Cardiac Arrest"),
                       tags$li("Cataracts"),
                       tags$li("Eczema"),
                       tags$li("Glaucoma"),
                       tags$li("Macular Degeneration"),
                     ),
                     tags$ul(
                       tags$li("Liver Damage"),
                       tags$li("Tuberculosis"),
                       tags$li("Paralysis"),
                       tags$li("Anemia"),
                       tags$li("Influenza"),
                       tags$li("Migraine"),
                       tags$li("Asthma"),
                       tags$li("Arthritis"),
                       tags$li("Conjunctivitis"),
                       tags$li("Melanoma"),
                       tags$li("Psoraisis"),
                     )
                 )
             ),
             
             div(class = "result-box",
                 h3("🔬 Technology Stack"),
                 tags$ul(
                   tags$li(strong("R Programming Language:"), " Statistical computing and analysis"),
                   tags$li(strong("Shiny Framework:"), " Interactive web application"),
                   tags$li(strong("RandomForest Package:"), " Machine learning classification"),
                   tags$li(strong("ggplot2:"), " Elegant Data visualization"),
                   tags$li(strong("DT Package:"), " Interactive data tables"),
                   tags$li(strong("caret Package:"), " Classificationa and Regression Training"),
                   tags$li(strong("e1071 Package:"), " Model training and evaluation based on statistics and probability"),
                   tags$li(strong("magick Package:"), "Advanced graphics and image processing"),
                   tags$li(strong("pdftools Package:"), " Text extraction, rendering and data conversion from PDF documents"),
                   tags$li(strong("stringr Package:"), " Pattern matching functions to detect, locate, extract, match, replace, and split strings"),                   
                   tags$li(strong("DT Package:"), " OCR based text extraction from images and pdfs")
                 )
             ),
             
             div(style = "background-color: #fee; padding: 20px; border-radius: 10px; 
                             border-left: 5px solid #dc3545; margin-top: 30px;",
                 h3("⚠️ IMPORTANT MEDICAL DISCLAIMER"),
                 p(strong("This is an educational AI demonstration tool and should NOT replace 
                       professional medical advice.")),
                 tags$ul(
                   tags$li("Always consult qualified healthcare professionals for diagnosis and treatment"),
                   tags$li("This tool is for informational and educational purposes only"),
                   tags$li("In case of emergency, immediately contact emergency services"),
                   tags$li("Do not use this tool as the sole basis for medical decisions"),
                   tags$li("Dietary plans should be reviewed by a registered dietitian"),
                   tags$li("Individual health conditions may require customized approaches")
                 )
             ),
             
             div(class = "result-box",
                 h3("👨‍💻 Development Information"),
                 p("This system has been developed as an educational demonstration of AI/ML 
                       applications in healthcare. It showcases how machine learning can be 
                       used to assist in preliminary health assessments and provide 
                       evidence-based dietary recommendations."),
                 p("Version: 2.0 Extended | Last Updated: 2025")
             )
         )
)
)
)



# Server
server <- function(input, output, session) {
  
  # Symptom-based Diagnosis
  diagnosis_results <- eventReactive(input$diagnose, {
    symptoms <- data.frame(
      Fatigue = as.integer(input$fatigue),
      Headache = as.integer(input$headache),
      Fever = as.integer(input$fever),
      Cough = as.integer(input$cough),
      Nausea = as.integer(input$nausea),
      Chest_Pain = as.integer(input$chest_pain),
      Joint_Pain = as.integer(input$joint_pain),
      Frequent_Urination = as.integer(input$freq_urination),
      Increased_Thirst = as.integer(input$thirst),
      Dizziness = as.integer(input$dizziness),
      Abdominal_Pain = as.integer(input$abdominal_pain),
      Shortness_of_Breath = as.integer(input$shortness_breath),
      Sleeplessness = as.integer(input$sleeplessness),
      Tiredness = as.integer(input$tiredness),
      Rapid_Hunger = as.integer(input$rapid_hunger),
      Loss_of_Appetite = as.integer(input$loss_appetite),
      Blurred_Vision = as.integer(input$blurred_vision),
      Weight_Loss = as.integer(input$weight_loss),
      Weakness = as.integer(input$tiredness),
      Numbness = as.integer(input$numbness),
      Speech_Difficulty = as.integer(input$confusion),
      Confusion = as.integer(input$confusion),
      Sweating = as.integer(input$sweating),
      Pale_Skin = as.integer(input$pale_skin),
      Jaundice = as.integer(input$jaundice),
      Night_Sweats = as.integer(input$sweating),
      Swelling = as.integer(input$swelling),
      Irregular_Heartbeat = as.integer(input$irregular_heartbeat),
      Eye_Pain = as.integer(input$eye_pain),
      Eye_Redness = as.integer(input$eye_redness),
      Light_Sensitivity = as.integer(input$light_sensitivity),
      Skin_Rash = as.integer(input$skin_rash),
      Itching = as.integer(input$itching),
      Skin_Discoloration = as.integer(input$pale_skin),
      Skin_Lesions = as.integer(input$skin_lesions)
    )
    
    result <- predict_disease(ml_model, symptoms)
    return(result)
  })
  
  output$diagnosis_output <- renderUI({
    req(input$diagnose)
    
    result <- tryCatch({
      diagnosis_results()
    }, error = function(e) {
      return(NULL)
    })
    
    if (is.null(result)) {
      return(div(class = "result-box",
                 h3("⚠️ Error in Analysis"),
                 p("Please select at least one symptom and try again.")))
    }
    
    # Get disease info for top prediction - with safety check
    disease_info <- NULL
    if (!is.null(result$disease) && result$disease %in% names(disease_info_db)) {
      disease_info <- disease_info_db[[result$disease]]
    }
    
    tagList(
      div(class = "result-box",
          h3("📋 Primary Diagnosis"),
          h2(style = "color: #28a745; font-weight: bold;", result$disease),
          h4(style = "color: #667eea;", paste0("Confidence Level: ", round(result$confidence, 2), "%")),
          br(),
          div(style = "background-color: #fff3cd; padding: 15px; border-radius: 8px;",
              p(strong("⚠️ Important:"), " This is an AI-based preliminary assessment. Please consult a healthcare professional for accurate diagnosis and treatment.")
          )
      ),
      
      div(class = "result-box",
          h3("📊 Complete Disease Probability Analysis"),
          p("This chart shows the likelihood of all diseases based on your selected symptoms:"),
          plotOutput("probability_plot", height = "600px")
      ),
      
      # Detailed information about the most probable disease
      if (!is.null(disease_info)) {
        tagList(
          div(class = "result-box",
              h3(paste("🔬 Detailed Information:", disease_info$full_name)),
              
              div(style = "background: #e7f3ff; padding: 20px; border-radius: 8px; margin: 15px 0;",
                  h4("📖 What is this condition?"),
                  p(style = "font-size: 16px; line-height: 1.8;", disease_info$description)
              ),
              
              div(style = "background: #fff3e0; padding: 20px; border-radius: 8px; margin: 15px 0;",
                  h4("🩺 Common Symptoms"),
                  div(style = "display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 10px;",
                      lapply(disease_info$symptoms, function(s) {
                        div(style = "background: white; padding: 10px; border-radius: 5px; border-left: 3px solid #ff9800;",
                            paste("•", s)
                        )
                      })
                  )
              ),
              
              div(style = "background: #ffebee; padding: 20px; border-radius: 8px; margin: 15px 0;",
                  h4("🔍 Possible Causes & Risk Factors"),
                  tags$ul(
                    style = "columns: 2; -webkit-columns: 2; -moz-columns: 2; font-size: 15px; line-height: 1.8;",
                    lapply(disease_info$causes, function(c) tags$li(c))
                  )
              ),
              
              div(style = "background: #e8f5e9; padding: 20px; border-radius: 8px; margin: 15px 0;",
                  h4("📌 Key Diagnostic Markers"),
                  div(style = "display: flex; flex-wrap: wrap; gap: 10px;",
                      lapply(disease_info$key_markers, function(m) {
                        span(style = "background: white; padding: 8px 15px; border-radius: 20px; 
                                     border: 2px solid #4caf50; font-weight: 600; color: #2e7d32;", m)
                      })
                  )
              ),
              
              div(style = "background: #f3e5f5; padding: 20px; border-radius: 8px; margin: 15px 0;",
                  h4("💊 Treatment & Management Options"),
                  tags$ol(
                    style = "font-size: 15px; line-height: 2;",
                    lapply(disease_info$treatment, function(t) tags$li(t))
                  )
              ),
              
              # Visual representation section
              div(style = "background: #e1f5fe; padding: 20px; border-radius: 8px; margin: 15px 0;",
                  h4("📸 Understanding the Condition"),
                  p("Below are conceptual representations to help you understand this condition:"),
                  div(style = "display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-top: 20px;",
                      # Symptom illustration card
                      div(style = "background: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);",
                          div(style = "width: 100%; height: 180px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                      border-radius: 8px; display: flex; align-items: center; justify-content: center;",
                              h2(style = "color: white; font-size: 60px;", "🩺")
                          ),
                          h5(style = "margin-top: 15px; color: #667eea;", "Common Symptoms"),
                          p(style = "font-size: 14px; color: #666;", 
                            paste("Key symptoms include:", paste(head(disease_info$symptoms, 3), collapse = ", ")))
                      ),
                      
                      # Affected area illustration
                      div(style = "background: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);",
                          div(style = "width: 100%; height: 180px; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                                      border-radius: 8px; display: flex; align-items: center; justify-content: center;",
                              h2(style = "color: white; font-size: 60px;", 
                                 ifelse(result$disease %in% c("Diabetes", "Hypertension", "Heart Attack", "Cardiac Arrest"), "❤️",
                                        ifelse(result$disease %in% c("Brain Stroke", "Paralysis", "Migraine"), "🧠",
                                               ifelse(result$disease %in% c("Cataracts", "Glaucoma", "Macular Degeneration", "Conjunctivitis"), "👁️",
                                                      ifelse(result$disease %in% c("Eczema", "Psoriasis", "Melanoma"), "🫱",
                                                             ifelse(result$disease %in% c("Asthma", "Common Cold", "Influenza", "Tuberculosis"), "🫁",
                                                                    ifelse(result$disease %in% c("Liver Damage"), "🫀",
                                                                           ifelse(result$disease %in% c("Gastritis"), "🫃", "🏥"))))))))
                          ),
                          h5(style = "margin-top: 15px; color: #f5576c;", "Affected Area"),
                          p(style = "font-size: 14px; color: #666;", 
                            ifelse(result$disease %in% c("Diabetes", "Hypertension", "Heart Attack", "Cardiac Arrest"), "Cardiovascular System",
                                   ifelse(result$disease %in% c("Brain Stroke", "Paralysis", "Migraine"), "Nervous System",
                                          ifelse(result$disease %in% c("Cataracts", "Glaucoma", "Macular Degeneration", "Conjunctivitis"), "Eyes",
                                                 ifelse(result$disease %in% c("Eczema", "Psoriasis", "Melanoma"), "Skin",
                                                        ifelse(result$disease %in% c("Asthma", "Common Cold", "Influenza", "Tuberculosis"), "Respiratory System",
                                                               ifelse(result$disease %in% c("Liver Damage"), "Liver",
                                                                      ifelse(result$disease %in% c("Gastritis"), "Digestive System", "Multiple Systems"))))))))
                      ),
                      
                      # Risk factors illustration
                      div(style = "background: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);",
                          div(style = "width: 100%; height: 180px; background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                                      border-radius: 8px; display: flex; align-items: center; justify-content: center;",
                              h2(style = "color: white; font-size: 60px;", "⚠️")
                          ),
                          h5(style = "margin-top: 15px; color: #4facfe;", "Risk Factors"),
                          p(style = "font-size: 14px; color: #666;", 
                            paste("Main causes:", paste(head(disease_info$causes, 2), collapse = ", ")))
                      ),
                      
                      # Treatment illustration
                      div(style = "background: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);",
                          div(style = "width: 100%; height: 180px; background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); 
                                      border-radius: 8px; display: flex; align-items: center; justify-content: center;",
                              h2(style = "color: white; font-size: 60px;", "💊")
                          ),
                          h5(style = "margin-top: 15px; color: #43e97b;", "Treatment Available"),
                          p(style = "font-size: 14px; color: #666;", 
                            paste("Options include:", paste(head(disease_info$treatment, 2), collapse = ", ")))
                      )
                  )
              )
          )
        )
      } else {
        div(class = "result-box",
            style = "background-color: #fff3cd;",
            h4("ℹ️ Limited Information Available"),
            p(paste("Please consult a healthcare professional for comprehensive information about these probable conditions."))
        )
      },
      
    )
  })
  
  output$probability_plot <- renderPlot({
    result <- diagnosis_results()
    probs <- as.data.frame(t(result$probabilities))
    probs$Disease <- rownames(probs)
    colnames(probs)[1] <- "Probability"
    
    # Sort by probability and show ALL diseases
    probs <- probs[order(-probs$Probability), ]
    
    # Color code: Top disease in green, next 4 in blue, rest in gray
    probs$Color <- ifelse(1:nrow(probs) == 1, "#28a745",
                          ifelse(1:nrow(probs) <= 5, "#667eea", "#cccccc"))
    
    ggplot(probs, aes(x = reorder(Disease, Probability), y = Probability * 100, fill = Color)) +
      geom_bar(stat = "identity", alpha = 0.9) +
      geom_text(aes(label = paste0(round(Probability * 100, 1), "%")), 
                hjust = -0.1, size = 3.5, fontface = "bold") +
      coord_flip() +
      scale_fill_identity() +
      labs(title = "Complete Disease Probability Distribution (All 23 Diseases)", 
           subtitle = "Green = Most Likely | Blue = Top 5 Candidates | Gray = Lower Probability",
           x = "", 
           y = "Probability (%)") +
      theme_minimal(base_size = 13) +
      theme(
        plot.title = element_text(hjust = 0.5, face = "bold", size = 16, color = "#333"),
        plot.subtitle = element_text(hjust = 0.5, size = 12, color = "#666"),
        axis.text.y = element_text(size = 11, color = "#333"),
        axis.text.x = element_text(size = 11),
        panel.grid.major.y = element_blank(),
        panel.grid.minor = element_blank(),
        panel.grid.major.x = element_line(color = "#e0e0e0")
      ) +
      ylim(0, max(probs$Probability * 100) * 1.15)
  })
  
  # Report Analysis
  report_analysis <- eventReactive(input$analyze_report, {
    req(input$report_file)
    
    file_path <- input$report_file$datapath
    file_ext <- tools::file_ext(input$report_file$name)
    
    report_text <- ""
    
    if (file_ext == "pdf") {
      report_text <- extract_text_from_pdf(file_path)
    } else if (file_ext %in% c("jpg", "jpeg", "png", "tiff")) {
      report_text <- extract_text_from_image(file_path)
    }
    
    analysis_result <- analyze_medical_report(report_text, disease_info_db)
    
    return(list(
      text = report_text, 
      findings = analysis_result$findings,
      detected_values = analysis_result$detected_values
    ))
  })
  
  output$report_analysis_output <- renderUI({
    req(input$analyze_report)
    analysis <- report_analysis()
    
    # Show extracted text
    extracted_text_ui <- div(class = "result-box",
                             h4("📄 Extracted Report Text"),
                             div(style = "background: white; padding: 15px; border: 1px solid #ddd; 
                                        border-radius: 5px; max-height: 300px; overflow-y: auto;",
                                 p(substr(analysis$text, 1, 2000)),
                                 if (nchar(analysis$text) > 2000) {
                                   p(em("... (text truncated for display)"))
                                 }
                             ))
    
    # Show detected lab values
    values_ui <- if (length(analysis$detected_values) > 0) {
      div(class = "result-box",
          h4("🔬 Detected Lab Values"),
          div(style = "display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 15px;",
              lapply(names(analysis$detected_values), function(test) {
                div(class = "info-card",
                    strong(test),
                    br(),
                    span(style = "color: #667eea; font-size: 18px; font-weight: bold;", 
                         analysis$detected_values[[test]])
                )
              })
          )
      )
    } else {
      div(class = "result-box", style = "background-color: #fff3cd;",
          h4("⚠️ No Lab Values Detected"),
          p("The system could not extract specific lab values from the uploaded report. This could be because:"),
          tags$ul(
            tags$li("The report is an image with poor quality"),
            tags$li("The report format is not standard"),
            tags$li("The OCR could not read the text accurately"),
            tags$li("The report doesn't contain numeric lab values")
          ),
          p(strong("Suggestion:"), " Try uploading a clearer image or a digital PDF report with text-based lab values.")
      )
    }
    
    # Show disease findings
    findings_ui <- if (length(analysis$findings) > 0) {
      lapply(names(analysis$findings), function(disease) {
        info <- disease_info_db[[disease]]
        
        if (is.null(info)) {
          return(div(class = "info-card",
                     h4(disease),
                     p(analysis$findings[[disease]]),
                     p(em("Note: Detailed disease information not available for this condition."))))
        }
        
        div(class = "result-box",
            h3(paste("🔬", info$full_name)),
            div(style = "background: #e7f3ff; padding: 15px; border-radius: 5px; margin-bottom: 15px;",
                p(strong("📊 Analysis:"), analysis$findings[[disease]])
            ),
            
            p(strong("Description:"), info$description),
            
            h5("📌 Standard Diagnostic Markers:"),
            div(style = "margin: 10px 0;",
                lapply(info$key_markers, function(m) {
                  span(class = "marker-badge", m)
                })
            ),
            
            h5("🔍 Common Causes:"),
            tags$ul(
              style = "columns: 2; -webkit-columns: 2; -moz-columns: 2;",
              lapply(info$causes, function(c) tags$li(c))
            ),
            
            h5("💊 Treatment Recommendations:"),
            tags$ul(
              lapply(info$treatment, function(t) tags$li(t))
            ),
            
            div(style = "background-color: #fff3cd; padding: 15px; border-radius: 5px; margin-top: 15px;",
                p(strong("⚠️ Important:"), 
                  " This analysis is based on automated text extraction. Always consult with your healthcare ",
                  "provider for accurate interpretation of your medical reports and appropriate treatment.")
            )
        )
      })
    }
    
    tagList(
      div(class = "result-box",
          h3("📋 Report Analysis Summary with Keywords Present In The Diagnositic Report"),
          p(strong("File analyzed:"), input$report_file$name),
          p(strong("Conditions detected:"), length(analysis$findings)),
          p(strong("Lab values extracted:"), length(analysis$detected_values))
      ),
      extracted_text_ui,
      values_ui,
      findings_ui
    )
  })
  
  # Dietary Plans
  diet_plan <- eventReactive(input$get_diet, {
    disease <- input$disease_select
    plan <- dietary_data[[disease]]
    return(list(plan = plan, disease = disease))
  })
  
  output$diet_output <- renderUI({
    req(input$get_diet)
    data <- diet_plan()
    plan <- data$plan
    
    tagList(
      h2(paste("🥗 Dietary Plan for", data$disease)),
      
      div(class = "result-box",
          h4("✅ Recommended Foods with Nutritional Values"),
          DTOutput("foods_table")
      ),
      
      div(class = "result-box",
          h4("❌ Foods to Avoid"),
          tags$ul(lapply(plan$foods_to_avoid, function(f) tags$li(f)))
      ),
      
      div(class = "result-box",
          h4("📅 Sample Daily Meal Plan"),
          DTOutput("meal_table")
      ),
      
      div(class = "result-box",
          h4("📈 Nutrition Summary"),
          tableOutput("nutrition_summary"),
          plotOutput("nutrition_chart", height = "300px")
      )
    )
  })
  
  output$foods_table <- renderDT({
    data <- diet_plan()
    datatable(data$plan$foods_to_eat, options = list(pageLength = 12), rownames = FALSE)
  })
  
  output$meal_table <- renderDT({
    data <- diet_plan()
    datatable(data$plan$sample_meal, options = list(pageLength = 10), rownames = FALSE)
  })
  
  output$nutrition_summary <- renderTable({
    data <- diet_plan()
    nutrition <- calculate_total_nutrition(data$plan$sample_meal)
    data.frame(
      Metric = c("Total Calories", "Total Protein", "Total Carbs", "Total Fat", "Total Fiber"),
      Value = c(paste(nutrition$Total_Calories, "kcal"),
                paste(nutrition$Total_Protein, "g"),
                paste(nutrition$Total_Carbs, "g"),
                paste(nutrition$Total_Fat, "g"),
                paste(nutrition$Total_Fiber, "g"))
    )
  })
  
  output$nutrition_chart <- renderPlot({
    data <- diet_plan()
    nutrition <- calculate_total_nutrition(data$plan$sample_meal)
    
    df <- data.frame(
      Nutrient = c("Protein", "Carbs", "Fat", "Fiber"),
      Grams = c(nutrition$Total_Protein, nutrition$Total_Carbs, 
                nutrition$Total_Fat, nutrition$Total_Fiber)
    )
    
    ggplot(df, aes(x = Nutrient, y = Grams, fill = Nutrient)) +
      geom_bar(stat = "identity") +
      geom_text(aes(label = paste0(Grams, "g")), vjust = -0.5) +
      labs(title = paste("Daily Nutrition -", nutrition$Total_Calories, "Calories")) +
      theme_minimal() +
      theme(legend.position = "none")
  })
}

shinyApp(ui = ui, server = server)


