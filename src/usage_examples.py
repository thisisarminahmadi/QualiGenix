"""
QualiGenix GenAI Agent - Usage Examples
Comprehensive examples of how to use the pharmaceutical AI assistant
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from genai_agent import QualiGenixAgent

def demonstrate_usage():
    """Demonstrate various ways to use the QualiGenix agent"""
    
    print("�� QualiGenix GenAI Agent - Usage Examples")
    print("="*60)
    
    # Initialize the agent
    agent = QualiGenixAgent()
    
    # Example 1: Basic Data Queries
    print("\n�� EXAMPLE 1: Basic Data Queries")
    print("-" * 40)
    
    queries = [
        "What is the average dissolution rate in our dataset?",
        "Show me batches with dissolution_av > 95",
        "What's the correlation between api_water and dissolution_av?",
        "Which batches have the highest batch_yield?"
    ]
    
    for query in queries:
        print(f"\n❓ Query: {query}")
        response = agent.query(query)
        print(f"�� Response: {response}")
    
    # Example 2: ML Predictions
    print("\n🔮 EXAMPLE 2: ML Predictions")
    print("-" * 40)
    
    prediction_queries = [
        "Predict dissolution_av for a batch with api_water=1.5, main_CompForce mean=4.2, tbl_speed_mean=100",
        "What would be the expected batch_yield with these parameters: api_content=94, tbl_fill_mean=5.3",
        "Estimate impurities_total for a batch with api_total_impurities=0.3, stiffness_mean=90"
    ]
    
    for query in prediction_queries:
        print(f"\n❓ Query: {query}")
        response = agent.query(query)
        print(f"🤖 Response: {response}")
    
    # Example 3: Process Optimization
    print("\n⚙️ EXAMPLE 3: Process Optimization")
    print("-" * 40)
    
    optimization_queries = [
        "How can we improve dissolution performance?",
        "What process parameters should we focus on to reduce impurities?",
        "How to optimize batch yield while maintaining quality?",
        "What are the critical factors for tablet hardness consistency?"
    ]
    
    for query in optimization_queries:
        print(f"\n❓ Query: {query}")
        response = agent.query(query)
        print(f"🤖 Response: {response}")
    
    # Example 4: Scenario Analysis
    print("\n📈 EXAMPLE 4: Scenario Analysis")
    print("-" * 40)
    
    scenario_queries = [
        "Analyze the scenario: We want to increase production speed by 20%",
        "What happens if we reduce compression force by 10%?",
        "How would changing raw material suppliers affect quality?",
        "Analyze the trade-offs of increasing tablet hardness"
    ]
    
    for query in scenario_queries:
        print(f"\n❓ Query: {query}")
        response = agent.query(query)
        print(f"🤖 Response: {response}")
    
    # Example 5: Regulatory Compliance
    print("\n�� EXAMPLE 5: Regulatory Compliance")
    print("-" * 40)
    
    compliance_queries = [
        "What are the regulatory requirements for dissolution testing?",
        "How do we ensure batch-to-batch consistency for FDA compliance?",
        "What documentation do we need for quality control?",
        "How to handle out-of-specification (OOS) results?"
    ]
    
    for query in compliance_queries:
        print(f"\n❓ Query: {query}")
        response = agent.query(query)
        print(f"�� Response: {response}")
    
    # Example 6: Troubleshooting
    print("\n🔧 EXAMPLE 6: Troubleshooting")
    print("-" * 40)
    
    troubleshooting_queries = [
        "Why did batch 15 fail dissolution testing?",
        "What caused the high impurity levels in batch 23?",
        "Why is our batch yield decreasing over time?",
        "How to fix tablet hardness variability?"
    ]
    
    for query in troubleshooting_queries:
        print(f"\n❓ Query: {query}")
        response = agent.query(query)
        print(f"🤖 Response: {response}")
    
    # Example 7: Advanced Analytics
    print("\n�� EXAMPLE 7: Advanced Analytics")
    print("-" * 40)
    
    analytics_queries = [
        "Perform a root cause analysis for quality issues",
        "Identify trends in our manufacturing data",
        "What are the key performance indicators for our process?",
        "How to implement statistical process control?"
    ]
    
    for query in analytics_queries:
        print(f"\n❓ Query: {query}")
        response = agent.query(query)
        print(f"🤖 Response: {response}")

def interactive_demo():
    """Interactive demo for testing custom queries"""
    print("\n🎯 INTERACTIVE DEMO")
    print("="*60)
    print("Enter your own questions about pharmaceutical manufacturing!")
    print("Type 'quit' to exit.")
    
    agent = QualiGenixAgent()
    
    while True:
        user_input = input("\n❓ Your question: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if user_input:
            print("🤖 QualiGenix is thinking...")
            response = agent.query(user_input)
            print(f"�� Response: {response}")

if __name__ == "__main__":
    # Run the demonstration
    demonstrate_usage()
    
    # Uncomment the line below to run interactive demo
    # interactive_demo() 