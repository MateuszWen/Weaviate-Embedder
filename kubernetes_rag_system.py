#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kubernetes RAG System with Weaviate
A RAG system for Kubernetes questions that automatically classifies questions
and directs them to the appropriate data sources (markdown for theory, csv for practice).
"""


import weaviate
import json
import re
import os
from typing import List, Dict, Any
from enum import Enum
from dotenv import load_dotenv


load_dotenv()


os.environ["OPENAI_API_KEY"] = ""


class QueryType(Enum):
    GENERAL_KNOWLEDGE = "general"  # questions about general knowledge -> "markdown"
    CLUSTER_STATUS = "cluster"     # questions about cluster status -> "csv"


class KubernetesRAGSystem:
    """Main class of the RAG system for Kubernetes"""
    
    def __init__(self, weaviate_url: str = "localhost", port: int = 8080):


        self.weaviate_url = weaviate_url
        self.port = port
        self.client = weaviate.connect_to_local(host=self.weaviate_url, port=self.port, grpc_port=50052)
        self.collection_name = "PromQLKnowledge"


        if self.client.is_live():
            print("Weaviate server is active.")
        else:
            print("Weaviate server is not responding.")


        
    def setup_collection(self):
        """Creates a collection in Weaviate"""
        from weaviate.classes.config import Configure, Property, DataType
        
        # Delete the collection if it exists
        try:
            self.client.collections.delete(self.collection_name)
            print(f"Deleted existing collection {self.collection_name}")
        except:
            pass
        
        # Creating a collection with embedding only for the "text" field
        self.client.collections.create(
            name=self.collection_name,
            vector_config=Configure.Vectors.text2vec_ollama(
                api_endpoint="http://host.docker.internal:11434",  # If using Docker you might need: (http://host.docker.internal:11434)
                model="nomic-embed-text",  # The model to use
                source_properties=["text"],  # only this field will be embedded
                name="text"
            ),
            generative_config=Configure.Generative.ollama(  # Configure the Ollama generative integration
                api_endpoint="http://host.docker.internal:11434",  # If using Docker you might need: (http://host.docker.internal:11434)
                model="llama3.1"  # The model to use
            ),
            properties=[
                Property(name="text", data_type=DataType.TEXT, skip_vectorization=True),
                Property(name="source", data_type=DataType.TEXT, skip_vectorization=True),
                Property(name="chunk_type", data_type=DataType.TEXT, skip_vectorization=True),
                Property(name="category", data_type=DataType.TEXT, skip_vectorization=True),
                Property(name="section", data_type=DataType.TEXT, skip_vectorization=True),
                Property(name="subsection", data_type=DataType.TEXT, skip_vectorization=True),
                Property(name="query", data_type=DataType.TEXT, skip_vectorization=True),
                Property(name="metadata_json", data_type=DataType.TEXT, skip_vectorization=True),
            ]
        )
        print(f"Created collection {self.collection_name}")
    
    def import_data(self, json_file_path: str):
        """Imports data from a JSON file to Weaviate"""
        collection = self.client.collections.get(self.collection_name)
        
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"Loaded {len(data)} records from {json_file_path}")
        
        print("Starting data import to Weaviate...")


        with collection.batch.fixed_size(batch_size=100) as batch:


            imported_count = 0
            for item in data:
                weaviate_object = {
                    "text": item.get("text", ""),
                    "source": item.get("source", ""),
                    "chunk_type": item.get("chunk_type", ""),
                    "category": item.get("category", ""),
                    "section": item.get("section", ""),
                    "subsection": item.get("subsection") or "",  # None -> empty string
                    "query": item.get("query") or "",  # None -> empty string
                    "metadata_json": json.dumps(item.get("metadata", {}), ensure_ascii=False)
                }
                
                batch.add_object(properties=weaviate_object)  #, vector=weaviate_object["text"])
                
                imported_count += 1
                
            if imported_count % 10 == 0:
                print(f"Imported {imported_count}/{len(data)} objects...")
        
        if len(collection.batch.failed_objects) > 0:
            print(f"Failed to import {len(collection.batch.failed_objects)} objects")


        self.client.close()  # Close the client after import  
        print(f"Successfully imported {int(imported_count) - len(collection.batch.failed_objects)} objects!")
    
    def classify_query_type(self, user_question: str) -> QueryType:
        """Classifies the user's question as general or about cluster status"""
        user_question_lower = user_question.lower()
        
        # Patterns for querying cluster status (specific metrics/monitoring)
        cluster_status_patterns = [
            r"ile\s+(podów|pod|nodes?|node|deployments?|svc|services?)",
            r"how many\s+(pods?|nodes?|deployments?|services?)",
            r"which\s+(pods?|nodes?|containers?).*(running|failed|ready)",
            r"jakie\s+(pody|nody|kontenery).*(działają|nie działają)",
            r"w\s+namespace",
            r"in\s+(namespace|production|staging|default)",
            r"na\s+nodzie",
            r"on\s+node",
            r"zużycie\s+(cpu|memory|pamięć)",
            r"(cpu|memory|disk)\s+(usage|utilization)",
            r"(restart|błęd|error|failed|crash)",
            r"status\s+(pod|deployment|node)",
            r"\b\d+\b.*(pod|node|replica)",  # questions with numbers
            r"więcej niż|less than|greater than|ponad",
            r"nginx|mysql|redis|api|backend|frontend|database"
        ]
        
        # Patterns for general knowledge questions
        general_knowledge_patterns = [
            r"czym jest|what is|what are|co to jest",
            r"jak działa|how does.*work|how to",
            r"różnica między|difference between",
            r"explain|wyjaśnij|opisz|describe",
            r"concept|pojęcie|architektura|architecture",
            r"best practices|najlepsze praktyki",
            r"(pod|service|deployment|node|namespace)\b(?!.*\b(in|w)\b.*\b(production|staging|namespace)\b)"
        ]
        
        # Check patterns for cluster status
        for pattern in cluster_status_patterns:
            if re.search(pattern, user_question_lower):
                return QueryType.CLUSTER_STATUS
                
        # Check patterns for general knowledge
        for pattern in general_knowledge_patterns:
            if re.search(pattern, user_question_lower):
                return QueryType.GENERAL_KNOWLEDGE
        
        # Default to treating as a cluster status question
        return QueryType.CLUSTER_STATUS
    
    def search_knowledge(self, query: str, query_type: QueryType, limit: int = 5) -> List[Dict[Any, Any]]:
        """Searches for relevant information from the database based on the question type"""
        collection = self.client.collections.get(self.collection_name)
        
        from weaviate.classes.query import Filter


        # Perform a vector search with a filter
        response = collection.query.near_text(
            query=query,
            limit=limit,
            # Select the appropriate filter based on the question type
            filters=(Filter.by_property("source").equal("markdown") if query_type == QueryType.GENERAL_KNOWLEDGE else Filter.by_property("source").equal("csv")),
            return_properties=["text", "source", "chunk_type", "category", "section", "subsection", "query", "metadata_json"]
        )
        
        # Process the results
        results = []
        for obj in response.objects:
            result = obj.properties.copy()
            # Parse metadata from JSON string
            try:
                result["metadata"] = json.loads(result["metadata_json"])
            except:
                result["metadata"] = {}
            del result["metadata_json"]
            results.append(result)
            
        return results
    
    def get_llm_context(self, user_question: str, limit: int = 5) -> Dict[str, Any]:
        """Main function to get context for the LLM"""
        # Classify the question
        query_type = self.classify_query_type(user_question)
        
        # Search for relevant information
        search_results = self.search_knowledge(user_question, query_type, limit)
        
        # Prepare the response for the LLM
        return {
            "user_question": user_question,
            "query_type": query_type.value,
            "source_type": "markdown" if query_type == QueryType.GENERAL_KNOWLEDGE else "csv",
            "results": search_results,
            "context_summary": f"Found {len(search_results)} results of type '{query_type.value}'"
        }


class KubernetesLLMIntegration:
    """Class integrating the RAG system with an LLM (OpenAI)"""
    
    def __init__(self, weaviate_url: str = "localhost", port: int = 8080):
        self.rag_system = KubernetesRAGSystem(weaviate_url, port)
        # OpenAI configuration from environment variable
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            print("WARNING: OPENAI_API_KEY is not set in environment variables!")
        
    def generate_response(self, user_question: str) -> Dict[str, Any]:
        """Generates an LLM response based on context from Weaviate"""
        # Get context from Weaviate
        context = self.rag_system.get_llm_context(user_question)
        
        # Prepare the prompt for the LLM
        system_prompt = self._build_system_prompt(context)
        
        # Actual OpenAI call (requires pip install openai)
        try:
            from openai import OpenAI
            openai = OpenAI(base_url="http://localhost:11434/v1", api_key=self.openai_api_key)
            
            response = openai.chat.completions.create(
                model="llama3.1:latest",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_question}
                ],
                temperature=0.1,
                max_tokens=1000
            )

            openai.close()
            
            return {
                "user_question": user_question,
                "llm_response": response.choices[0].message.content,
                "context_used": context,
                "sources_count": len(context["results"]),
                "source_type": context["source_type"]
            }
            
        except Exception as e:
            openai.close()
            return {
                "user_question": user_question,
                "error": f"LLM Error: {str(e)}",
                "context_used": context
            }
    
    def _build_system_prompt(self, context: Dict[str, Any]) -> str:
        """Builds the system prompt based on the question type and context"""
        if context["source_type"] == "markdown":
            prompt = """You are a Kubernetes expert who answers general questions about concepts and architecture.
                        You base your answers on the provided information from the documentation.
                        Context from the knowledge base:
                        """
        else:
            prompt = """You are a Kubernetes monitoring expert who helps create PromQL queries and analyze cluster status.
                        You base your answers on the provided examples of PromQL queries.
                        Context from the knowledge base (PromQL queries):
                        """
        
        # Add results from Weaviate
        for i, result in enumerate(context["results"][:3], 1):  # max 3 results
            prompt += f"\n{i}. Category: {result['category']}"
            prompt += f"\n   Section: {result['section']}"
            if result.get('query'):
                prompt += f"\n   PromQL Query: {result['query']}"
            prompt += f"\n   Description: {result['text'][:300]}...\n"
        
        action_type = "If the question is about a specific cluster state, provide the appropriate PromQL query" if context['source_type'] == 'csv' else 'Explain the concepts clearly and understandably'
        
        prompt += f"""\nYour tasks:
                1. Answer the user's question based ONLY on the provided context
                2. {action_type}
                3. If you do not have enough information, state it clearly
                4. Respond in English
                5. Be precise and specific"""
        
        return prompt


def main():
    """Test function"""
    print("=== KUBERNETES RAG SYSTEM TEST ===\n")
    
    # System initialization
    llm_integration = KubernetesLLMIntegration()
    
    # Example test questions
    test_questions = [
        "What is a pod in Kubernetes?",  # markdown
        "How many pods are running in the production namespace?",  # csv
        "Which containers are using the most CPU in production?",  # csv
        "How does a Service work in Kubernetes?",  # markdown
        "What are the best practices for a Deployment?"  # markdown
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"{i}. QUESTION: {question}")
        print("-" * 80)
        
        result = llm_integration.generate_response(question)
        
        print(f"Source Type: {result.get('source_type', 'N/A')}")
        print(f"Number of Sources: {result.get('sources_count', 0)}")
        print(f"Question Type: {result.get('context_used', {}).get('query_type', 'N/A')}")
        
        if 'llm_response' in result:
            print(f"\nLLM Response:\n{result['llm_response']}")
        elif 'simulated_response' in result:
            print(f"\nSimulated Response:\n{result['simulated_response']}")
        elif 'error' in result:
            print(f"\nError: {result['error']}")
        
        print("\n" + "="*100 + "\n")


def setup_system():
    """Function for system setup - run once at the beginning"""
    print("=== SYSTEM SETUP ===")
    
    # Check if the data file exists
    json_file = "promql_chunks.json"
    if not os.path.exists(json_file):
        print(f"ERROR: File not found {json_file}")
        print("Make sure the data file is in the same directory.")
        return False
    
    # RAG system initialization
    rag_system = KubernetesRAGSystem()
    
    try:
        # Create collection
        print("1. Creating collection in Weaviate...")
        rag_system.setup_collection()
        
        # Import data
        print("2. Importing data...")
        rag_system.import_data(json_file)
        
        print("\n✅ System configured successfully!")
        print("You can now run main() for testing.")
        return True
        
    except Exception as e:
        print(f"❌ Error during setup: {str(e)}")
        print("Check if Weaviate is running at http://localhost:8080")
        rag_system.client.close()
        return False


if __name__ == "__main__":
    # Comment out the line below after the first run
    setup_system()  # Run this once at the beginning
    
    # System testing
    main()
