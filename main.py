import re
import os
import json
import os.path
import sqlite3
import datetime
import numpy as np
import pandas as pd
from io import StringIO
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv(override=True)
# print(data.head())
print(os.getenv("AZURE_OPENAI_API_KEY"))
print(os.getenv("AZURE_OPENAI_ENDPOINT"))

force = eval(os.getenv("FORCE"))

# Create an in-memory SQLite database
conn = sqlite3.connect(':memory:')  # ':memory:' creates an in-memory database

# Display the resulting DataFrame
# print(string_columns_df)

# print(unique_values)
class NL2SQL:
    def __init__(self, dict_path_csv={"analytical_table":'analytical_table.csv'},encoding="latin-1",separator=",",has_header=True,ignore_errors=True):
        
        # self.data = {table_name: pl.read_csv(dict_path_csv[table_name], ignore_errors=ignore_errors, has_header=has_header, separator=separator, encoding=encoding) for table_name in dict_path_csv}
        self.data = {table_name: pd.read_csv(dict_path_csv[table_name], sep=separator, encoding=encoding) for table_name in dict_path_csv}
        self.string_columns_df = {table_name: self.data[table_name].select_dtypes(include=['object']) for table_name in dict_path_csv}
        # self.string_columns_df = {table_name: self.data[table_name].select(pl.col(pl.String)) for table_name in dict_path_csv}
        # self.data.select(pl.col(pl.String))
        self.file_name = datetime.datetime.now().strftime("logs/%Y-%m-%d_%H:%M:%S")
        directory = os.path.dirname(self.file_name)

        if not os.path.exists(directory):
            os.makedirs(directory)

        f = open(f"{self.file_name}.json", "w")
        json.dump({'iterations':[]}, f)
        f.close()
        f = open(f"{self.file_name}.txt", "w")
        f.close()

    def get_filename(self) -> str:
        return self.file_name

    # def filter_values(self, column: str, value: str):
    #     return self.data.filter(pl.col(column).str.contains(repr(value)))

    def get_attributes(self, table_name:str) -> list:
        return list(self.data[table_name].columns.tolist()
)

    def filter_values(self, column: str, value: str):
        return {table_name: df[df[column].str.contains(value, na=False)] for table_name, df in self.data.items()}

    
    # def get_sample(self, table_name:str, n: int = 5) -> str:
    #     return f"{self.data[table_name].sample(n=n, seed=0).write_csv()}"

    def get_sample(self, table_name: str, n: int = 5) -> str:
        sample_df = self.data[table_name].sample(n=n, random_state=0)  # Sample n rows with a fixed random state
        output = StringIO()  # Create an in-memory text stream
        sample_df.to_csv(output, index=False)  # Write the DataFrame to the stream without the index
        return output.getvalue()  # Return the CSV content as a string

    def get_sample_prompt(self, number_of_samples: int) -> list:
        prompts = {}
        for table in self.data:
            prompt = f"""Given the table sample below:

Table name: {table}
{self.get_sample(table,number_of_samples)}

Create a list of with a description of the the table and attribute in the following format:
    Table name; natural language description of the table
    Attribute name; natural language description of the table; attribute type; format; primary key

where:
    Attribute type contains the basic type of the attribute.
    Format contains a description of the value format (e.g. date format, separators, etc).
    Primary key: a boolean true | false in case the attribute is likely to be a primary key.

Important: 
    Must do it for all attributes. Just return the list.
    Do not return the original table sample."""
            prompts[table] = prompt
            
        return prompts
    
    def get_unique_values(self) -> dict:
        unique_values = {}
        for table_name in self.string_columns_df:
            unique_values[table_name] = {}
            for col in self.string_columns_df[table_name].columns:
                unique_values[table_name][col] = self.string_columns_df[table_name][col].unique()
        return unique_values
        # return {table_name: {col: self.string_columns_df[table_name][col].unique()} for table_name in self.string_columns_df for col in self.string_columns_df[table_name].columns}
    

    def get_prompt_joins(self, tables: list, descriptions:dict, number_of_samples: int) -> str:
        prompt = f"List the natural join between the two tables below.\n\n"

        for i in range(0, len(tables), 2):  # Iterate through the indices of the tables list in steps of 2
            t1 = tables[i]  # First table in the pair
            prompt += f"{descriptions[t1]}\n{self.get_sample(t1, number_of_samples)}\n\n"
            
            if i + 1 < len(tables):  # Check if there's a second table to process
                t2 = tables[i + 1]  # Second table in the pair
                prompt += f"{descriptions[t2]}\n{self.get_sample(t2, number_of_samples)}\n\n"

        # for t in tables:
            # table = self.data[t]

            # prompt += f"{descriptions[t]}\n{self.get_sample(t,number_of_samples)}\n\n"

        return f"{prompt}\n\nUse the following format:\nTable1 name.attribute name ; differences in value format"

    def get_prompt_relevant_tables_and_attributes(self, nl_query:str, descriptions:dict):
        prompt = f"""Select the relevant tables and attributes given the natural language query below. Return only the list of tables and attributes.
{nl_query} 
And the set of tables:
"""
        for table in descriptions:
            prompt += f"{descriptions[table]}\n\n"
        return prompt
    
    def get_prompt_relevant_tables_and_attributes_table_filter(self, nl_query:str, descriptions:dict, tables:str):
        prompt = f"""Select the relevant tables and attributes given the natural language query below. Return only the list of tables and attributes.
{nl_query} 
And the set of tables:
"""
        for table in descriptions:
            if table in tables:
                prompt += f"{descriptions[table]}\n\n"
        return prompt
    
    def get_prompt_get_instances(self, nl_query:str):
        prompt = f"""Given the following natural language query to be mapped to an SQL query:
{nl_query} 
Determine the set of references to values in the table, i.e. terms which are likely to map to VALUES in a WHERE = ‘VALUE’ clause."""
        return prompt
    
    def get_prompt_summary_prompt(self, schema_description:str):
        prompt = f"""Given the schema description below, provide a summary description of the table limited to 3 sentences.
{schema_description}
"""
        return prompt
    
    def get_prompt_relevant_tables(self, nl_query:str, list_of_tables:str):
        prompt = f"""Select from the list of tables below, the tables which are relevant to answer the following natural language query: {nl_query} 

Instructions:
Just return the list of table names.

List of Tables:
{list_of_tables}
"""
        return prompt
    
    def get_prompt_correct_sqlquery(self, error:str):
        prompt = f"""Correct this sql query:
{error}

Ouput only the new sql query"""
        return prompt



    def get_prompt_nl_to_sql(self,nl_query:str,saida3:str,saida4:str,joins,num_examples:int=3):
        query_plan_context = f"{saida3}\n{saida4}\n"
        should_add_where = True
        unique_values = self.get_unique_values()
        for table in unique_values:
            print(f"Keys: {unique_values[table].keys()}")
            for column in unique_values[table]:
                if (table.strip() in saida4 or table.strip() in saida3) and (column.strip() in saida3 or column.strip() in saida4):
                    if should_add_where:
                        query_plan_context += "Here are examples of the style of data stored in the database:\n"
                        should_add_where = False
                    # print(unique_values[table][column])
                    # print(','.join(np.random.choice(unique_values[table][column], 2, replace=False)))
                    query_plan_context += f"{table}.{column}: {','.join(np.random.choice(unique_values[table][column], num_examples if len(unique_values[table][column])>=num_examples else len(unique_values[table][column]), replace=False))}\n"
        query_plan_context += f"{joins}\n"
        prompt = f"""Given the natural language query: 
{nl_query}
and the set of relevant tables, attributes, target values and joins in:
{query_plan_context}
Write the corresponding SQL query.
Build the SQL query in a step-wise manner:
    Determine the table and the projection attributes.
    Determine the select statement clauses.
    Determine the joins.
    Define the aggregation operations (if applicable).
Return only the SQL query."""
        return prompt
    
    def map_value_instace(self,relevant_tables_attributes:dict):
        unique_values = self.get_unique_values()
        values_allowed = {}
        for table in relevant_tables_attributes:
            if table in unique_values:
                for attr in relevant_tables_attributes[table]:
                    if attr in unique_values[table]:
                        for u in unique_values[table][attr]:
                            if relevant_tables_attributes[table][attr].lower() == u.lower():
                                if table not in values_allowed:
                                    values_allowed[table] = {}
                                if attr not in values_allowed[table]:
                                    values_allowed[table][attr] = []
                                values_allowed[table]



    def generate(self,prompt:str):    
        client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
            api_version="2024-02-01",
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        )
    
        deployment_name='lunar-chatgpt-4o'

        response = client.chat.completions.create(model=deployment_name, messages=[
            {"role": "user", "content": prompt},
        ])

        self.write_log(input=prompt, output=response.choices[0].message.content.strip())

        return response.choices[0].message.content.strip()
    
    def generate_list_dict(self,prompt:list):    
        client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-02-01",
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        )
    
        deployment_name='lunar-chatgpt-4o' #This will correspond to the custom name you chose for your deployment when you deployed a model. Use a gpt-35-turbo-instruct deployment. 
        
        
        # Send a completion call to generate an answer
        response = client.chat.completions.create(model=deployment_name, messages=prompt)
        input_log = ''
        for p in prompt:
            input_log += '{ ' + f"role: {p['role']}, content: {p['content']}" + ' }\n'
        self.write_log(input=input_log, output=response.choices[0].message.content.strip())
        
        return response.choices[0].message.content.strip()
    
    def check_all_columns(self, model_reponse:str, table:str) -> list:
        missing = []
        for c in self.data[table].columns:
            c = c.strip()
            if c not in model_reponse:
                missing.append(c)
        return missing

    def write_log(self, input:str, output:str):

        with open(f"{self.file_name}.json", "r") as f:
            data = json.load(f)

        data['iterations'].append({"input": f"{input}"})
        data['iterations'].append({"output": f"{output}"})

        with open(f"{self.file_name}.json", "w") as f:
            json.dump(data, f)

        with open(f"{self.file_name}.txt", "a") as f:
            f.write(f"\ninput:\n{input}\n---------------------------------\n")

        with open(f"{self.file_name}.txt", "a") as f:
            f.write(f"output:\n{output}\n---------------------------------\n")



def read_questions_from_file(question_file_path):
    with open(question_file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        
        questions = re.findall(r"'(.*?)'", content, re.DOTALL)
        
    return questions


with open(os.getenv("TABLE_DICT"),"r") as f:
    dict_path_csv = json.load(f)

# Write the DataFrame to the SQLite database
for csv_file_name in dict_path_csv:
    df = pd.read_csv(dict_path_csv[csv_file_name], encoding="utf8")
    df.to_sql(csv_file_name, conn, index=False, if_exists='replace')

question_file_path = os.getenv("QUESTION_FILE_PATH")
print(question_file_path)
questions = read_questions_from_file(question_file_path)
question_log_file = os.getenv("QUESTION_LOG_FILE_PATH")
print(question_log_file)

if not os.path.isdir(question_log_file):
    os.makedirs(question_log_file)

for nl_query in questions:
    # break
    nl_query = nl_query[0].upper() + nl_query[1:]

    if os.path.isfile(f"{question_log_file}/{nl_query}.txt"):
        continue

    obj = NL2SQL(dict_path_csv=dict_path_csv)
    print(nl_query)
    desciption_file = f"{question_file_path.split('.')[0]}_description.json"
    # if not os.path.isfile(desciption_file) or (force and False):
    if not os.path.isfile(desciption_file):
        description = {}

        for table in dict_path_csv:
            prompt = [
                {"role": "user", "content": obj.get_sample_prompt(5)[table]},
            ]
            description[table] = obj.generate_list_dict(prompt)
            missing = obj.check_all_columns(description[table], table)
            if len(missing):
                prompt.append({"role": "assistant", "content": description[table]})
                prompt.append({"role": "user", "content": f"You are missing: {','.join(missing)}\nWrite only about the columns not already described"})
            # print(prompt)
            # description[table] = f"{description[table]}\n\n{obj.generate_list_dict(prompt)}"
        f = open(desciption_file, "w")
        json.dump(description, f)
        f.close()
    else:
        print("ENTREI AQUI - description")
        with open(desciption_file, "r") as f:
            description = json.load(f)
        for table in dict_path_csv:
            if table not in description:
                prompt = [
                    {"role": "user", "content": obj.get_sample_prompt(5)[table]},
                ]
                description[table] = obj.generate_list_dict(prompt)
                missing = obj.check_all_columns(description[table], table)
                if len(missing):
                    prompt.append({"role": "assistant", "content": description[table]})
                    prompt.append({"role": "user", "content": f"You are missing: {','.join(missing)}\nWrite only about the columns not already described"})

            else:
                obj.write_log(input=obj.get_sample_prompt(5)[table], output=description[table])

        f = open(desciption_file, "w")
        json.dump(description, f)
        f.close()

    # break

    posible_joins_file = f"{question_file_path.split('.')[0]}_posible_joins.json"
    posible_joins = {}
    posible_joins["table1_table2"] = ""
    # if len(dict_path_csv)>1:
    #     if not os.path.isfile(posible_joins_file) or force and False:
    #         posible_joins["table1_table2"] = obj.generate(obj.get_prompt_joins(list(dict_path_csv.keys()), description, 5))
    #         f = open(posible_joins_file, "w")
    #         json.dump(posible_joins, f)
    #         f.close()
    #     else:
    #         print("ENTREI AQUI - posible_joins")
    #         with open(posible_joins_file, "r") as f:
    #             posible_joins = json.load(f)
    #         obj.write_log(input=obj.get_prompt_joins(list(dict_path_csv.keys()), description, 5), output=posible_joins["table1_table2"])


    summary_file = f"{question_file_path.split('.')[0]}_table_summary.json"
    # if not os.path.isfile(summary_file) or (force and False):
    if not os.path.isfile(summary_file):
        table_summary = {}
        for table in dict_path_csv:
            table_summary[f"{table}"] = obj.generate(obj.get_prompt_summary_prompt(description[table]))
        f = open(summary_file, "w")
        json.dump(table_summary, f)
        f.close()
    else:
        print("ENTREI AQUI - table_summary")
        with open(summary_file, "r") as f:
            table_summary = json.load(f)
        for table in dict_path_csv:
            if table not in table_summary:
                table_summary[f"{table}"] = obj.generate(obj.get_prompt_summary_prompt(description[table]))
            else:
                obj.write_log(input=obj.get_prompt_summary_prompt(description[table]), output=table_summary[table])
            
        f = open(summary_file, "w")
        json.dump(table_summary, f)
        f.close()

    # break

    list_of_tables = ""

    table_items = list(dict_path_csv.items())

    for i in range(0, len(table_items), 20):

        table_batch = table_items[i:i + 20]
        
        for table_name, table_path in table_batch:
            list_of_tables += f"Table name: {table_name}\n"
            list_of_tables += f"Description: {description[table_name]}\n"
            list_of_tables += f"Attributes: {', '.join(obj.get_attributes(table_name))}\n\n"

        if not os.path.isfile("relevant_table.json") or (force):
            relevant_table = {}
            relevant_table[i] = obj.generate(obj.get_prompt_relevant_tables(nl_query,list_of_tables))
            f = open("relevant_table.json", "w")
            json.dump(relevant_table, f)
            f.close()
        else:
            print("ENTREI AQUI - relevant_table")
            with open("relevant_table.json", "r") as f:
                relevant_table = json.load(f)

    # break

    if not os.path.isfile("saida3.json") or force:
        saida3 = {}
        saida3[nl_query] = obj.generate(obj.get_prompt_relevant_tables_and_attributes_table_filter(nl_query = nl_query, descriptions = description, tables="\n".join(list(relevant_table.values()))))
        f = open("saida3.json", "w")
        json.dump(saida3, f)
        f.close()
    else:
        print("ENTREI AQUI - saida3")
        with open("saida3.json", "r") as f:
            saida3 = json.load(f)

    # break

    if not os.path.isfile("saida4.json") or force:
        saida4 = {}
        prompt_chat = [
                {"role": "user", "content": obj.get_prompt_relevant_tables_and_attributes_table_filter(nl_query = nl_query, descriptions = description, tables=saida3[nl_query])},
                {"role": "assistant", "content": saida3[nl_query]},
                {"role": "user", "content": obj.get_prompt_get_instances(nl_query=nl_query)}
            ]
        saida4[nl_query] = obj.generate_list_dict(prompt_chat)
        f = open("saida4.json", "w")
        json.dump(saida4, f)
        f.close()
    else:
        print("ENTREI AQUI - saida4")
        with open("saida4.json", "r") as f:
            saida4 = json.load(f)
        

    if not os.path.isfile("saida5.json") or force:
        saida5 = {}
        # for table in dict_path_csv:
        saida5[nl_query] = obj.generate(obj.get_prompt_nl_to_sql(nl_query=nl_query,saida3=saida3[nl_query],saida4=saida4[nl_query],joins=posible_joins["table1_table2"]))
        f = open("saida5.json", "w")
        json.dump(saida5, f)
        f.close()
    else:
        print("ENTREI AQUI")
        with open("saida5.json", "r") as f:
            saida5 = json.load(f)
    
    with open(f"{question_log_file}/{nl_query}.txt","w") as answer:
        answer.write(f"{nl_query}\n\n")
        answer.write("-"*50)
        answer.write("\n")
        answer.write(str(saida5[nl_query]))
    # break

    query_failed = False
    try:
        query = re.findall(r"^(?!.*\bsql\b)[\s\S]+?;", saida5[nl_query].replace("```","").strip()+";", re.MULTILINE)[0]
        # with open("predict_imdb.txt", "a") as fff:
        #     backslash = "\n"
        #     fff.write(f"{query.replace(backslash,' ')}{backslash}")

        result = pd.read_sql(query, conn)
        print(result)
    except Exception as e:
        result = e
        query_failed = True
        print(result)


    if query_failed:
        query_failed_correction = {}
        # for table in dict_path_csv:
        query_failed_correction[nl_query] = obj.generate(obj.get_prompt_correct_sqlquery(result))
        f = open("query_failed_correction.json", "w")
        json.dump(query_failed_correction, f)
        f.close()
        result = query_failed_correction[nl_query]
        try:
            query = re.findall(r"^(?!.*\bsql\b)[\s\S]+?;", query_failed_correction[nl_query].strip()+";", re.MULTILINE)[0]
            # with open("predict_imdb.txt", "a") as fff:
            #     backslash = "\n"
            #     fff.write(f"{query.replace(backslash,' ')}{backslash}")
            result = pd.read_sql(query, conn)
            print(result)
        except Exception as e:
            result = e
            query_failed = True
            print(result)

    with open(os.getenv("PREDICT_FILE_PATH"), "a") as fff:
        backslash = "\n"
        fff.write(f"{query.replace(backslash,' ')}{backslash}")

                
    with open(f"{question_log_file}/{nl_query}.txt","w") as answer:
        answer.write(f"{nl_query}\n\n")
        answer.write("-"*50)
        answer.write("\n")
        answer.write(str(saida5[nl_query]))


    with open(f"{obj.get_filename()}.txt","a") as answer:
        answer.write("\n")
        answer.write("-"*50)
        answer.write("\n")
        answer.write(str(result))
    with open(f"question_last/{nl_query}.txt","w") as answer:
        answer.write(f"{nl_query}\n\n")
        answer.write("-"*50)
        answer.write("\n")
        answer.write(str(result))
    break
# print(obj.data)

conn.close()