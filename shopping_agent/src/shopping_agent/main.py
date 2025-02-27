from crewai.flow.flow import Flow, start, listen, router
from litellm import completion
from dotenv import load_dotenv
import json

load_dotenv()

class ShoppingAgent(Flow):
    model = "gemini/gemini-2.0-flash"
    
    inventory = json.load(open("inventory.json"))
    cart = []

    @start()
    def retrieve_user_query(self):
        '''Retrieve the user query'''

        print("Welcome to the shopping agent")
        user_query = input("Enter your shopping query: ")
        
        response = completion(
            model=self.model,
            messages=[
                {"role": "user", "content": f"You are a shopping assistant. 
                You are given a user query {user_query}. 
                Process the user query and return whether the user is looking to search for a product, compare products, or manage a shopping cart (add/view items)."}
            ]
        )   
        query_type = response["choices"][0]["message"]["content"]
        return query_type
    
    @router(retrieve_user_query)
    def route_query(self, query_type):
        '''Route the user query to the appropriate function'''

        if "search" in query_type.lower():
            return 'search'
        elif "compare" in query_type.lower():
            return 'compare'
        elif "cart" in query_type.lower():
            return 'cart'
        else:
            return 'unknown'
        
    @listen("search")
    def search_product(self, user_query):
        '''Search for the product that best matches the user query'''

        response = completion(
            model=self.model,
            messages=[
                {"role": "user", "content": f"You are a shopping assistant. You are given a user query {user_query}. Search the {self.inventory} for the product that best matches the user query."}
            ]
        )
        product_name = response["choices"][0]["message"]["content"]
        print(f"🔍 Found product: {product_name}")
        
        add_to_cart = input(f"Would you like to add '{product_name}' to your cart? (yes/no): ").strip().lower()
        if add_to_cart == "yes":
            self.cart.append(product_name)

        return f"✅ '{product_name}' has been added to your cart!"
    
    @listen("compare")
    def compare_products(self, user_query):
        '''Compare the products according the features listed in the inventory and recommend the best product'''

        response = completion(
            model=self.model,
            messages=[
                {"role": "user", "content": f"You are a shopping assistant. You are given a user query {user_query}. Compare the products according the features listed in the {self.inventory} and recommend the best product."}
            ]
        )
        return response["choices"][0]["message"]["content"]
    
    @listen("cart")
    def view_cart(self, user_query=None):
        '''Display the current items in the cart'''
        
        if not self.cart:
            return "Your cart is empty."
        
        return f"Your cart contains: {', '.join(self.cart)}"
    
    @listen("unknown")
    def unknown_query(self, user_query):
        '''Handle unknown user queries'''
        
        response = completion(
            model=self.model,
            messages=[
                {"role": "user", "content": f"You are a shopping assistant. You are given a user query {user_query}. The user query is not clear. Please ask the user to provide more information."}
            ]
        )
        return response["choices"][0]["message"]["content"]

def kickoff():
    agent = ShoppingAgent()
    result = agent.kickoff()
    print(f"Results: {result}")
