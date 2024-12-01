import os
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_openai import ChatOpenAI
import boto3
from botocore.config import Config
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage


# Individual Tools for specific AWS operations
class S3BucketListTool(BaseTool):
    name = "list_s3_buckets"
    description = "Lists all S3 bucket names in the AWS account"
    s3_client: Any = None

    def __init__(self, s3_client: Any):
        super().__init__()
        self.s3_client = s3_client

    def _run(self) -> List[str]:
        response = self.s3_client.list_buckets()
        return [bucket['Name'] for bucket in response['Buckets']]

class S3BucketAccessTool(BaseTool):
    name = "check_bucket_access"
    description = "Checks if S3 buckets are public or private"
    s3_client: Any = None

    def __init__(self, s3_client: Any):
        super().__init__()
        self.s3_client = s3_client

    def _run(self, bucket_name: str) -> Dict:
        try:
            policy = self.s3_client.get_bucket_policy_status(Bucket=bucket_name)
            acl = self.s3_client.get_bucket_acl(Bucket=bucket_name)
            is_public = policy['PolicyStatus']['IsPublic']
            return {
                "bucket_name": bucket_name,
                "is_public": is_public,
                "acl_details": acl
            }
        except Exception as e:
            return {"bucket_name": bucket_name, "error": str(e)}

class S3BucketFileCountTool(BaseTool):
    name = "get_bucket_file_count"
    description = "Gets the number of files in each S3 bucket"
    s3_client: Any = None

    def __init__(self, s3_client: Any):
        super().__init__()
        self.s3_client = s3_client

    def _run(self, bucket_name: str) -> Dict:
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            count = 0
            for page in paginator.paginate(Bucket=bucket_name):
                if 'Contents' in page:
                    count += len(page['Contents'])
            return {"bucket_name": bucket_name, "file_count": count}
        except Exception as e:
            return {"bucket_name": bucket_name, "error": str(e)}

class S3BucketFileListTool(BaseTool):
    name = "list_bucket_files"
    description = "Lists all files in a specific S3 bucket"
    s3_client: Any = None

    def __init__(self, s3_client: Any):
        super().__init__()
        self.s3_client = s3_client

    def _run(self, bucket_name: str) -> Dict:
        try:
            files = []
            paginator = self.s3_client.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=bucket_name):
                if 'Contents' in page:
                    files.extend([obj['Key'] for obj in page['Contents']])
            return {"bucket_name": bucket_name, "files": files}
        except Exception as e:
            return {"bucket_name": bucket_name, "error": str(e)}

class EC2InstanceInfoTool(BaseTool):
    name = "get_ec2_instance_info"
    description = "Gets information about an EC2 instance by IP"
    ec2_client: Any = None

    def __init__(self, ec2_client: Any):
        super().__init__()
        self.ec2_client = ec2_client

    def _run(self, ip_address: str) -> Dict:
        try:
            # Search for instances with the given IP
            filters = [
                {'Name': 'private-ip-address', 'Values': [ip_address]},
                {'Name': 'public-ip-address', 'Values': [ip_address]}
            ]
            response = self.ec2_client.describe_instances(Filters=filters)

            if response['Reservations']:
                instance = response['Reservations'][0]['Instances'][0]
                return {
                    "instance_id": instance['InstanceId'],
                    "instance_type": instance['InstanceType'],
                    "state": instance['State']['Name'],
                    "launch_time": str(instance['LaunchTime'])
                }
            return {"error": "No instance found with the given IP"}
        except Exception as e:
            return {"error": str(e)}

class IAMUserPermissionsTool(BaseTool):
    name = "get_iam_user_permissions"
    description = "Gets permissions for a specific IAM user"
    iam_client: Any = None

    def __init__(self, iam_client: Any):
        super().__init__()
        self.iam_client = iam_client

    def _run(self, username: str) -> Dict:
        try:
            permissions = {
                "user_policies": [],
                "group_policies": [],
                "attached_policies": []
            }

            # Get user's inline policies
            user_policies = self.iam_client.list_user_policies(UserName=username)
            permissions["user_policies"] = user_policies['PolicyNames']

            # Get user's attached policies
            attached_policies = self.iam_client.list_attached_user_policies(UserName=username)
            permissions["attached_policies"] = [p['PolicyName'] for p in attached_policies['AttachedPolicies']]

            # Get user's groups and their policies
            groups = self.iam_client.list_groups_for_user(UserName=username)
            for group in groups['Groups']:
                group_policies = self.iam_client.list_group_policies(GroupName=group['GroupName'])
                permissions["group_policies"].extend(group_policies['PolicyNames'])

            return permissions
        except Exception as e:
            return {"error": str(e)}

class AWSToolkit:
    def __init__(self, aws_access_key: str, aws_secret_key: str, aws_region: str):
        self.config = Config(region_name=aws_region)
        self.session = boto3.Session(
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=aws_region
        )

        # Initialize AWS clients
        self.s3_client = self.session.client('s3', config=self.config)
        self.ec2_client = self.session.client('ec2', config=self.config)
        self.iam_client = self.session.client('iam', config=self.config)

        # Initialize tools
        self.tools = [
            S3BucketListTool(self.s3_client),
            S3BucketAccessTool(self.s3_client),
            S3BucketFileCountTool(self.s3_client),
            S3BucketFileListTool(self.s3_client),
            EC2InstanceInfoTool(self.ec2_client),
            IAMUserPermissionsTool(self.iam_client)
        ]

class IntelligentAWSAssistant:
    def __init__(self, openai_api_key: str, aws_access_key: str, aws_secret_key: str, aws_region: str):
        # Validation and initialization (similar to previous script)
        if not openai_api_key or not aws_access_key or not aws_secret_key:
            raise ValueError("API keys must be provided")

        self.llm = ChatOpenAI(
            temperature=0.3, 
            openai_api_key=openai_api_key, 
            model="gpt-3.5-turbo"
        )
        
        self.toolkit = AWSToolkit(aws_access_key, aws_secret_key, aws_region)

        # Create a more specific system prompt
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=(
                "You are a specialized AWS resource assistant focused on six specific tasks:\n"
                "1. List S3 bucket names\n"
                "2. Check if S3 buckets are public or private\n"
                "3. Count files in each S3 bucket\n"
                "4. List file names in each bucket\n"
                "5. Get EC2 instance details by IP\n"
                "6. Retrieve IAM user permissions\n\n"
                "If a question is outside these areas, politely inform the user about the scope of your capabilities."
            )),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
            ("human", "{input}")
        ])

        # Create the agent
        self.agent = create_openai_tools_agent(
            llm=self.llm, 
            tools=self.toolkit.tools, 
            prompt=self.prompt
        )

        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.toolkit.tools,
            verbose=True,
            max_iterations=3
        )

        # Replace question_categories with question_categories
        self.question_categories = {
            "bucket_list": {
                "description": "List S3 bucket names",
                "examples": [
                    "What S3 buckets do I have?",
                    "Show me all my buckets",
                    "List my S3 buckets"
                ]
            },
            "bucket_access": {
                "description": "Check S3 bucket public/private status",
                "examples": [
                    "Which buckets are public?",
                    "Is my bucket XYZ private?",
                    "Show me bucket access settings"
                ]
            },
            "file_count": {
                "description": "Count files in each S3 bucket",
                "examples": [
                    "How many files are in bucket XYZ?",
                    "Count files in my buckets",
                    "Show file count for each bucket"
                ]
            },
            "file_list": {
                "description": "List file names in each bucket",
                "examples": [
                    "What files are in bucket XYZ?",
                    "Show me the contents of bucket XYZ",
                    "List all files in my bucket"
                ]
            },
            "ec2_info": {
                "description": "Get EC2 instance details by IP",
                "examples": [
                    "What's the size of EC2 instance with IP 1.2.3.4?",
                    "Give me details about EC2 instance with IP 1.2.3.4",
                    "Show EC2 instance information"
                ]
            },
            "iam_permissions": {
                "description": "Retrieve IAM user permissions",
                "examples": [
                    "What permissions does user XYZ have?",
                    "Show me user XYZ's access rights",
                    "List permissions for IAM user"
                ]
            }
        }


    def categorize_question(self, question: str) -> str:
        """Categorizes the user's question using the LLM"""
        categorization_prompt = f"""
        Given the following question categories and their examples:

        {self._format_categories_for_prompt()}

        Please categorize this question: "{question}"

        Return only the category key (e.g., 'bucket_list', 'bucket_access', etc.).
        If the question doesn't match any category, return 'unknown'.

        Category:"""

        response = self.llm.predict(categorization_prompt).strip().lower()
        return response if response in self.question_categories else "unknown"

    def _format_categories_for_prompt(self) -> str:
        """Formats the categories and their examples for the prompt"""
        formatted = []
        for cat_key, cat_data in self.question_categories.items():
            formatted.append(f"{cat_key}:")
            formatted.append(f"Description: {cat_data['description']}")
            formatted.append("Examples:")
            for example in cat_data['examples']:
                formatted.append(f"- {example}")
            formatted.append("")
        return "\n".join(formatted)

    def process_question(self, question: str) -> str:
        """Process user questions and return appropriate responses"""
        try:
            # Execute the agent with the question
            response = self.agent_executor.invoke({
                "input": question
            })

            # Return the output or a default message
            return response.get('output', 'I could not find a specific answer to your question.')
        
        except Exception as e:
            return f"Error processing your question: {str(e)}"

    def _format_response(self, response: str, category: str) -> str:
        """Formats the response based on the category and adds relevant context"""
        formatted_response = f"Category: {self.question_categories[category]['description']}\n\n"
        formatted_response += response
        return formatted_response

def main():
    try:
        # Credentials
        openai_api_key = "..."
        aws_access_key = "..."
        aws_secret_key = "..."
        aws_region = "..."

        print("\n🤖 Initializing AWS Intelligent Assistant...")
        assistant = IntelligentAWSAssistant(
            openai_api_key=openai_api_key,
            aws_access_key=aws_access_key,
            aws_secret_key=aws_secret_key,
            aws_region=aws_region
        )

        print("\n✨ AWS Assistant is ready!")
        print("\n💡 You can ask questions about:")
        for category, data in assistant.question_categories.items():
            print(f"\n📌 {data['description']}:")
            for example in data['examples']:
                print(f"   - {example}")

        print("\n❔ Type 'exit' to quit\n")

        while True:
            question = input("\nYou: ").strip()

            if question.lower() == 'exit':
                print("\n👋 Goodbye! Have a great day!")
                break

            if not question:
                print("Please ask a question!")
                continue

            print("\n🤔 Processing...")
            try:
                response = assistant.process_question(question)
                print(f"\nAssistant: {response}")
            except Exception as e:
                print(f"\n❌ Error: Something went wrong while processing your question.")
                print(f"Details: {str(e)}")
                print("Please try rephrasing your question or check if it's related to supported categories.")

    except KeyboardInterrupt:
        print("\n\n👋 Program interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ Fatal Error: {str(e)}")
        print("Please check your API keys and AWS credentials.")
        print("Make sure you have the required permissions to access AWS services.")

if __name__ == "__main__":
    main()