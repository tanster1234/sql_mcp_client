import streamlit as st
import asyncio
import anthropic
import textwrap
from mcp import ClientSession
from mcp.client.sse import sse_client
from typing import Union, cast, List, Dict, Any
import pandas as pd
import io
import logging
import uuid
import json
import re
import plotly.express as px
import plotly.graph_objects as go
import os
import dotenv
import codecs


def extract_relevant_tables(user_query: str, schema_info: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Returns only the schema info for tables mentioned in the user query."""
    query_lower = user_query.lower()
    relevant_tables = []
    
    for table in schema_info:
        full_name = f"{table['schema']}.{table['table']}".lower()
        short_name = table['table'].lower()

        if full_name in query_lower or short_name in query_lower:
            relevant_tables.append(table)
    
    return relevant_tables



class PostgreSQLAssistantApp:
    def __init__(self):
        st.set_page_config(
            page_title="Enterprise PostgreSQL Assistant",
            page_icon="🔍",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # Load environment variables
        dotenv.load_dotenv()
        self.db_url = os.getenv('DATABASE_URL')
        self.pg_mcp_url = os.getenv('PG_MCP_URL', 'https://159d-104-13-14-137.ngrok-free.app/sse')
        self.anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')

        if not self.anthropic_api_key:
            st.error("ANTHROPIC_API_KEY environment variable is not set.")
            st.stop()

        if 'messages' not in st.session_state:
            st.session_state.messages = []

        if 'selectbox_keys' not in st.session_state:
            st.session_state.selectbox_keys = set()

        if 'last_query_result' not in st.session_state:
            st.session_state.last_query_result = ""

        if 'sql_finished' not in st.session_state:
            st.session_state.sql_finished = False

        if 'conn_id' not in st.session_state:
            st.session_state.conn_id = None

        if 'schema_info' not in st.session_state:
            st.session_state.schema_info = []

        self.anthropic_client = anthropic.AsyncAnthropic(
            api_key=self.anthropic_api_key
        )

    async def fetch_schema_info(self, session, conn_id):
        """Fetch database schema information from the MCP server."""
        schema_info = []
        
        try:
            # First get all schemas
            schemas_resource = f"pgmcp://{conn_id}/schemas"
            schemas_response = await session.read_resource(schemas_resource)
            
            schemas_content = None
            if hasattr(schemas_response, 'content') and schemas_response.content:
                schemas_content = schemas_response.content
            elif hasattr(schemas_response, 'contents') and schemas_response.contents:
                schemas_content = schemas_response.contents
                
            if schemas_content:
                content = schemas_content[0]
                if hasattr(content, 'text'):
                    schemas = json.loads(content.text)
                    
                    # For each schema, get its tables
                    for schema in schemas:
                        schema_name = schema.get('schema_name')
                        schema_description = schema.get('description', '')
                        
                        # Fetch tables for this schema
                        tables_resource = f"pgmcp://{conn_id}/schemas/{schema_name}/tables"
                        tables_response = await session.read_resource(tables_resource)
                        
                        tables_content = None
                        if hasattr(tables_response, 'content') and tables_response.content:
                            tables_content = tables_response.content
                        elif hasattr(tables_response, 'contents') and tables_response.contents:
                            tables_content = tables_response.contents
                            
                        if tables_content:
                            content = tables_content[0]
                            if hasattr(content, 'text'):
                                tables = json.loads(content.text)
                                
                                # For each table, get its columns
                                for table in tables:
                                    table_name = table.get('table_name')
                                    table_description = table.get('description', '')
                                    
                                    # Fetch columns for this table
                                    columns_resource = f"pgmcp://{conn_id}/schemas/{schema_name}/tables/{table_name}/columns"
                                    columns_response = await session.read_resource(columns_resource)
                                    
                                    columns = []
                                    columns_content = None
                                    if hasattr(columns_response, 'content') and columns_response.content:
                                        columns_content = columns_response.content
                                    elif hasattr(columns_response, 'contents') and columns_response.contents:
                                        columns_content = columns_response.contents
                                        
                                    if columns_content:
                                        content = columns_content[0]
                                        if hasattr(content, 'text'):
                                            columns = json.loads(content.text)
                                    
                                    # Add table with its columns to schema info
                                    schema_info.append({
                                        'schema': schema_name,
                                        'table': table_name,
                                        'description': table_description,
                                        'columns': columns
                                    })
            
            return schema_info
        except Exception as e:
            st.error(f"Error fetching schema information: {e}")
            logging.error(f"Error fetching schema information: {e}")
            return []

    def format_schema_for_prompt(self, schema_info):
        """Format schema information as a string for the prompt."""
        if not schema_info:
            return "No schema information available."
        
        schema_text = "DATABASE SCHEMA:\n\n"
        
        for table_info in schema_info:
            schema_name = table_info.get('schema')
            table_name = table_info.get('table')
            description = table_info.get('description', '')
            
            schema_text += f"Table: {schema_name}.{table_name}"
            if description:
                schema_text += f" - {description}"
            schema_text += "\n"
            
            columns = table_info.get('columns', [])
            if columns:
                schema_text += "Columns:\n"
                for col in columns:
                    col_name = col.get('column_name', '')
                    data_type = col.get('data_type', '')
                    is_nullable = col.get('is_nullable', '')
                    description = col.get('description', '')
                    
                    schema_text += f"  - {col_name} ({data_type}, nullable: {is_nullable})"
                    if description:
                        schema_text += f" - {description}"
                    schema_text += "\n"
            
            schema_text += "\n"
        
        return schema_text

    def get_unique_key(self, prefix=''):
        while True:
            key = f"{prefix}_{uuid.uuid4().hex[:8]}"
            if key not in st.session_state.selectbox_keys:
                st.session_state.selectbox_keys.add(key)
                return key

    def render_header(self):
        st.markdown("""
        # 📂 Enterprise PostgreSQL Intelligence
        ### Advanced Data Query & Insights Platform
        """)
        st.divider()

    def render_sidebar(self):
        with st.sidebar:
            st.header("🛠️ Query Configuration")

            # Database URL input
            if not self.db_url:
                self.db_url = st.text_input("PostgreSQL Connection URL", 
                                         placeholder="postgresql://pguser:pgpass@local-postgres:5432/pgdb",
                                         type="password")

            model_key = self.get_unique_key('model')
            model = st.selectbox(
                "Select AI Model", 
                ["claude-3-7-sonnet-latest", "claude-3-5-sonnet-20240620"],
                key=model_key
            )

            tokens_key = self.get_unique_key('tokens')
            max_tokens = st.slider(
                "Max Response Tokens", 
                min_value=1000, 
                max_value=16000, 
                value=10000,
                key=tokens_key
            )

            st.divider()
            st.info("""
            💡 Pro Tip:
            - Use clear, precise SQL queries
            - Check table names before querying
            - Leverage AI for complex data analysis
            """)

            # Display database connection status
            if st.session_state.conn_id:
                st.success(f"Connected to database")
            elif self.db_url:
                st.warning("Not connected to database")

        return model, max_tokens

    async def generate_visualizations(self, model):
        result_text = st.session_state.last_query_result
        if not result_text:
            st.warning("No final output available for visualization.")
            return

        system_prompt = """
        You are a data visualization expert. You will receive the result of a SQL query in plain text (not a DataFrame).
        It may contain insights or summaries, not necessarily tabular data. Your task is to propose meaningful and
        relevant visualizations using plotly based on the textual result. Only return Python code that creates the visualizations.
        
        IMPORTANT:
        1. NEVER use .get() method on any object that might be undefined or null
        2. Always check if a variable exists before accessing its properties
        3. Use defensive programming practices: check if dataframes are empty before plotting
        4. Ensure all plotly operations have proper error handling
        5. If the input data isn't suitable for visualization, return a simple message saying so
        """

        messages = [
            {"role": "user", "content": f"Here is the SQL query result:\n\n{result_text}\n\nPlease generate visualizations if appropriate."}
        ]

        try:
            response = await self.anthropic_client.messages.create(
                model=model,
                system=system_prompt,
                max_tokens=2000,
                messages=messages
            )

            for i, content in enumerate(response.content):
                if content.type == "text":
                    st.code(content.text, language='python')
                    try:
                        cleaned_code = re.sub(r'^```(?:python)?\n|```$', '', content.text.strip(), flags=re.MULTILINE).strip()
                        
                        # Create safe execution environment with error handling
                        exec_globals = {
                            "st": st,
                            "pd": pd,
                            "px": px,
                            "go": go,
                            "result_text": result_text
                        }
                        
                        # Safe DataFrame creation function
                        def safe_create_dataframe(data, **kwargs):
                            try:
                                return pd.DataFrame(data, **kwargs)
                            except Exception as df_err:
                                st.warning(f"Failed to create DataFrame: {df_err}")
                                return pd.DataFrame()  # Return empty dataframe instead of None
                        
                        exec_globals['safe_create_dataframe'] = safe_create_dataframe
                        
                        # Patch timeline to avoid x_start == x_end error
                        original_timeline = px.timeline
                        def safe_timeline(*args, **kwargs):
                            try:
                                df = kwargs.get('data_frame', args[0] if args else None)
                                if df is not None and 'x_start' in kwargs and 'x_end' in kwargs:
                                    if isinstance(df, pd.DataFrame) and not df.empty:  # Check if df is not empty
                                        df = df.copy()
                                        x_start = kwargs['x_start']
                                        x_end = kwargs['x_end']
                                        if x_start in df.columns and x_end in df.columns:  # Verify columns exist
                                            # Safely check for equal values
                                            equal_values = df[x_start] == df[x_end]
                                            if isinstance(equal_values, pd.Series) and equal_values.any():
                                                df[x_end] = pd.to_datetime(df[x_end]) + pd.Timedelta(days=1)
                                        kwargs['data_frame'] = df
                                fig = original_timeline(*args, **kwargs)
                                fig.show = lambda: st.plotly_chart(fig, use_container_width=True)
                                return fig
                            except Exception as timeline_err:
                                st.warning(f"Timeline plot error: {timeline_err}")
                                # Return empty figure on error
                                empty_fig = go.Figure()
                                empty_fig.show = lambda: st.plotly_chart(empty_fig, use_container_width=True)
                                return empty_fig
                        
                        exec_globals['px'].timeline = safe_timeline
                        
                        # Add a safe figure creation wrapper
                        def safe_figure(*args, **kwargs):
                            try:
                                fig = go.Figure(*args, **kwargs)
                                fig.show = lambda: st.plotly_chart(fig, use_container_width=True)
                                return fig
                            except Exception as fig_err:
                                st.warning(f"Figure creation error: {fig_err}")
                                empty_fig = go.Figure()
                                empty_fig.show = lambda: st.plotly_chart(empty_fig, use_container_width=True)
                                return empty_fig
                        
                        exec_globals['safe_figure'] = safe_figure
                        
                        # Patch all show() methods to use Streamlit and add error handling
                        def patched_show(fig):
                            try:
                                st.plotly_chart(fig, use_container_width=True)
                            except Exception as show_err:
                                st.warning(f"Error displaying chart: {show_err}")
                        
                        exec_globals['go'].Figure.show = patched_show
                        
                        # Safe patching for other plotly functions
                        safe_functions = ['pie', 'scatter', 'bar', 'line', 'histogram', 'box']
                        for func_name in safe_functions:
                            if hasattr(px, func_name):
                                original_func = getattr(px, func_name)
                                def safe_plot_func(original=original_func):
                                    def wrapper(*args, **kwargs):
                                        try:
                                            fig = original(*args, **kwargs)
                                            fig.show = lambda: st.plotly_chart(fig, use_container_width=True)
                                            return fig
                                        except Exception as plot_err:
                                            st.warning(f"Plot creation error: {plot_err}")
                                            empty_fig = go.Figure()
                                            empty_fig.show = lambda: st.plotly_chart(empty_fig, use_container_width=True)
                                            return empty_fig
                                    return wrapper
                                exec_globals['px'].__dict__[func_name] = safe_plot_func()
                        
                        # Process the text result to create a DataFrame if needed
                        def safe_process_text_to_df(text):
                            try:
                                # Try to parse as JSON first
                                try:
                                    data = json.loads(text)
                                    if isinstance(data, list):
                                        return pd.DataFrame(data)
                                    elif isinstance(data, dict):
                                        return pd.DataFrame([data])
                                except json.JSONDecodeError:
                                    pass
                                
                                # Try to parse as CSV
                                try:
                                    return pd.read_csv(io.StringIO(text))
                                except:
                                    pass
                                
                                # Try to extract table-like data with line splitting
                                lines = text.strip().split('\n')
                                if len(lines) > 1:
                                    # Check if it looks like a table with headers
                                    header = lines[0]
                                    if '|' in header or '\t' in header or ',' in header:
                                        delimiter = '|' if '|' in header else ('\t' if '\t' in header else ',')
                                        return pd.read_csv(io.StringIO(text), delimiter=delimiter)
                                
                                # Just return empty DataFrame if all else fails
                                return pd.DataFrame()
                            except Exception as process_err:
                                st.warning(f"Failed to process text to DataFrame: {process_err}")
                                return pd.DataFrame()
                        
                        exec_globals['safe_process_text_to_df'] = safe_process_text_to_df
                        
                        # Add the function to the globals
                        exec_globals['result_df'] = safe_process_text_to_df(result_text)
                        
                        # Wrap the execution in a try-except block for additional safety
                        try:
                            exec(cleaned_code, exec_globals)
                        except Exception as exec_err:
                            st.error(f"Visualization code execution error: {exec_err}")
                            
                            # Try to simplify the code and run a basic visualization as fallback
                            if isinstance(exec_globals.get('result_df'), pd.DataFrame) and not exec_globals['result_df'].empty:
                                st.write("Attempting simplified visualization:")
                                df = exec_globals['result_df']
                                
                                # Create a simple bar or line chart based on dataframe structure
                                try:
                                    if len(df.columns) >= 2:
                                        numeric_cols = df.select_dtypes(include=['number']).columns
                                        if len(numeric_cols) > 0:
                                            # Get the first numeric column for a simple bar chart
                                            y_col = numeric_cols[0]
                                            x_col = df.columns[0] if df.columns[0] != y_col else df.columns[1]
                                            
                                            fig = px.bar(df, x=x_col, y=y_col, title=f"Simple visualization of {y_col} by {x_col}")
                                            st.plotly_chart(fig, use_container_width=True)
                                except Exception as fallback_err:
                                    st.warning(f"Fallback visualization failed: {fallback_err}")
                    
                    except Exception as exec_err:
                        st.error(f"Execution error: {exec_err}")

        except Exception as e:
            st.warning(f"Could not generate visualizations: {e}")


    async def generate_sql_with_anthropic(self, user_query, schema_text, model, max_tokens):
        """Generate SQL using Claude with response template prefilling."""
        
        system_prompt = f"""You are an expert PostgreSQL developer who will translate a natural language query into a SQL query.

Before executing any query, first verify the table names and structure. 
If tables are missing, explain why the query cannot be executed.
You must provide your response in JSON format with two required fields:
1. "explanation": A brief explanation of your approach to the query
2. "sql": The valid, executable PostgreSQL SQL query

IMPORTANT: If your SQL query contains curly braces {{ or }}, you must escape them by doubling them: {{ becomes {{ and }} becomes }}.

Here is the database schema you will use:
{schema_text}
"""
        
        try:
            # Use response template prefilling to force Claude to produce JSON
            response = await self.anthropic_client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_query},
                    {"role": "assistant", "content": '{"explanation": "'}
                ]
            )
            
            # Extract the result
            result_text = response.content[0].text
            
            # First try to parse the complete JSON response
            try:
                if not result_text.strip().endswith('}'):
                    result_text += '}'
                result_json = json.loads(result_text)
                
                # If we have both fields, process the SQL to handle escaped braces
                if "explanation" in result_json and "sql" in result_json:
                    # Replace escaped braces in SQL
                    sql = result_json["sql"].replace("{{", "{").replace("}}", "}")
                    result_json["sql"] = sql
                    return result_json
            except json.JSONDecodeError:
                # If parsing failed, try to extract fields manually
                pass
                
            # Manual extraction fallback
            explanation = ""
            sql = ""
            
            # Extract explanation
            if '"explanation":' in result_text:
                explanation_part = result_text.split('"explanation":', 1)[1].strip()
                if explanation_part.startswith('"'):
                    explanation = explanation_part.split('"', 2)[1]
                else:
                    explanation = explanation_part.split(',', 1)[0].strip()
                    if explanation.endswith('"'):
                        explanation = explanation[:-1]
            
            # Extract SQL
            if '"sql":' in result_text:
                sql_part = result_text.split('"sql":', 1)[1].strip()
                if sql_part.startswith('"'):
                    sql = sql_part.split('"', 2)[1]
                else:
                    # Find the end of the SQL statement, looking for the last non-JSON closing brace
                    parts = sql_part.split('}')
                    if len(parts) > 1:
                        # If there are multiple closing braces, take everything except the last one
                        # as the last one is likely the JSON closing brace
                        sql = '}'.join(parts[:-1]).strip()
                    else:
                        sql = sql_part.strip()
                    if sql.endswith('"'):
                        sql = sql[:-1]
                
                # Replace escaped braces in SQL
                sql = sql.replace("{{", "{").replace("}}", "}")
            
            return {
                "explanation": explanation.replace('{"explanation": "', ''),
                "sql": sql
            }
                
        except Exception as e:
            st.error(f"Error calling Anthropic API: {e}")
            import traceback
            st.error(traceback.format_exc())
            return {
                "explanation": f"Error: {str(e)}",
                "sql": ""
            }

    async def process_query(self, session, query, model, max_tokens):
        try:
            # Add user message to chat history
            with st.chat_message("user"):
                st.write(query)
                
            st.session_state.messages.append({"role": "user", "content": query})
            
            # Get available tools from the server
            response = await session.list_tools()
            available_tools = [
                {
                    "name": tool.name,
                    "description": tool.description or "",
                    "input_schema": tool.inputSchema,
                }
                for tool in response.tools
            ]

            # Force a new connection for each query with strict validation
            conn_id = None
            if self.db_url:
                connection_placeholder = st.empty()
                with connection_placeholder.container():
                    with st.status("Establishing database connection...") as status:
                        try:
                            # Explicitly clear the previous connection ID
                            st.session_state.conn_id = None
                            
                            # Use the connect tool to register a new connection
                            connect_result = await session.call_tool(
                                "connect",
                                {
                                    "connection_string": self.db_url
                                }
                            )
                            
                            # Wait a moment to ensure connection is established
                            await asyncio.sleep(1)
                            
                            # Extract connection ID with strict validation
                            conn_id = None
                            if hasattr(connect_result, 'content') and connect_result.content:
                                for content_item in connect_result.content:
                                    if hasattr(content_item, 'text'):
                                        try:
                                            result_data = json.loads(content_item.text)
                                            if isinstance(result_data, dict) and 'conn_id' in result_data:
                                                conn_id = result_data.get('conn_id')
                                                break
                                        except json.JSONDecodeError:
                                            continue
                            
                            if not conn_id:
                                status.update(label="Failed: Could not obtain valid connection ID", state="error")
                                st.error("Failed to obtain a valid connection ID. Please check your database connection.")
                                return
                            
                            # Store the connection ID in session state
                            st.session_state.conn_id = conn_id
                            logging.info(f"Established connection with ID: {conn_id}")
                            status.update(label=f"Connection established successfully. ID: {conn_id[:8]}...")
                            
                            # Update status for schema fetching
                            status.update(label="Fetching database schema...")
                            
                            # Explicitly wait for schema information with a timeout and retry logic
                            max_retries = 3
                            for attempt in range(1, max_retries + 1):
                                try:
                                    # Use asyncio.wait_for to add a timeout
                                    schema_info = await asyncio.wait_for(
                                        self.fetch_schema_info(session, conn_id),
                                        timeout=20  # 20 second timeout for schema fetching
                                    )
                                    
                                    # Only update schema info if we got valid data
                                    if schema_info:
                                        st.session_state.schema_info = schema_info
                                        status.update(label=f"Schema loaded successfully with {len(schema_info)} tables")
                                        break
                                    else:
                                        if attempt < max_retries:
                                            status.update(label=f"Schema empty, retrying ({attempt}/{max_retries})...")
                                            await asyncio.sleep(2)  # Wait before retry
                                        else:
                                            status.update(label="Warning: Schema loaded but empty", state="warning")
                                            logging.warning("Schema information returned empty after all retries")
                                except asyncio.TimeoutError:
                                    if attempt < max_retries:
                                        status.update(label=f"Schema fetch timed out, retrying ({attempt}/{max_retries})...")
                                        await asyncio.sleep(2)  # Wait before retry
                                    else:
                                        status.update(label="Warning: Schema fetch timed out, using existing schema", state="warning")
                                        logging.warning("Schema fetch timed out after all retries")
                                except Exception as schema_error:
                                    if attempt < max_retries:
                                        status.update(label=f"Schema fetch error, retrying ({attempt}/{max_retries})...")
                                        logging.error(f"Schema fetch error (attempt {attempt}): {schema_error}")
                                        await asyncio.sleep(2)  # Wait before retry
                                    else:
                                        status.update(label=f"Error fetching schema: {str(schema_error)}", state="error")
                                        logging.error(f"Schema fetch error (final): {schema_error}")
                            
                        except Exception as e:
                            status.update(label=f"Connection failed: {str(e)}", state="error")
                            st.error(f"Failed to establish a connection: {e}")
                            logging.error(f"Connection error: {e}")
                            return  # Don't proceed if we can't establish a connection
                
                # Clear the connection status to save space after it's done
                connection_placeholder.empty()

            # Additional verification that connection ID exists before proceeding
            if not st.session_state.conn_id:
                st.error("No connection ID available. Cannot execute query.")
                return

            # Verify conn_id matches what we expect
            if conn_id and st.session_state.conn_id != conn_id:
                logging.error(f"Connection ID mismatch: local={conn_id}, session={st.session_state.conn_id}")
                st.error("Connection ID inconsistency detected. Updating to latest.")
                st.session_state.conn_id = conn_id

            # Display the connection ID for debugging
            st.info(f"Using connection ID: {st.session_state.conn_id[:8]}... (truncated for security)")

            # Get schema information for the prompt
            # relevant_schema = extract_relevant_tables(query, st.session_state.schema_info)
            schema_text = self.format_schema_for_prompt(st.session_state.schema_info)
            # print(relevant_schema)
            print(schema_text)
            
            system_prompt = f"""You are a master PostgreSQL assistant with deep knowledge of SQL and database operations.
            Before executing any query, you should use the provided schema context for the tables and fields.
            The schema context has already been supplied and includes detailed column comments that provide context about each field and table comments about each table.
            Do not re-fetch or verify table structures from the database. You already have the necessary schema information to execute the queries accurately.

            If tables are missing or if there are issues with the query, explain why the query cannot be executed based on the provided schema.
            You are expected to leverage this context to generate more accurate and meaningful queries without querying the database for schema information.

            Your job is to use the tools at your disposal to execute SQL queries and provide the results to the user.
            If a query fails, analyze the error and try a corrected version, making use of the schema context.

            IMPORTANT: When using the pg_query tool, you must always provide both the query and conn_id parameters.
            The current conn_id is: {st.session_state.conn_id}
            Always use this exact conn_id value when making pg_query calls.
            Never pass the connection_string to the pg_query tool.

            Here is the database schema with column comments:
            {schema_text}
            """

            messages = st.session_state.messages

            while True:
                # Generate SQL using Claude with tools available
                ai_response = await self.anthropic_client.messages.create(
                    model=model,
                    system=system_prompt,
                    max_tokens=max_tokens,
                    messages=messages,
                    tools=available_tools
                )

                assistant_message_content = []
                tool_uses = []

                for content in ai_response.content:
                    if content.type == "text":
                        assistant_message_content.append({"type": "text", "text": content.text})
                        with st.chat_message("assistant"):
                            st.write(content.text)
                    elif content.type == "tool_use":
                        tool_uses.append(content)
                        assistant_message_content.append({
                            "type": "tool_use",
                            "id": content.id,
                            "name": content.name,
                            "input": content.input
                        })

                messages.append({
                    "role": "assistant",
                    "content": assistant_message_content
                })

                if not tool_uses:
                    break

                # Recheck connection ID before executing tools
                if not st.session_state.conn_id:
                    st.error("Connection ID lost during query execution. Please try again.")
                    break

                tool_results = []
                for tool_use in tool_uses:
                    try:
                        # For pg_query tool, ensure we have conn_id
                        if tool_use.name == "pg_query":
                            # Double-check connection ID availability
                            if not st.session_state.conn_id:
                                tool_results.append({
                                    "type": "tool_result",
                                    "tool_use_id": tool_use.id,
                                    "content": "Error: No active database connection. Please connect first."
                                })
                                continue
                            
                            # Ensure the input has query parameter
                            tool_input = cast(dict, tool_use.input)
                            if "query" not in tool_input:
                                tool_results.append({
                                    "type": "tool_result",
                                    "tool_use_id": tool_use.id,
                                    "content": "Error: Query parameter is missing"
                                })
                                continue
                            
                            # Create a fresh copy of the input to avoid modifying the original
                            tool_input = dict(tool_input)
                            
                            # Check if conn_id is already in the input and log it
                            if "conn_id" in tool_input:
                                existing_conn_id = tool_input["conn_id"]
                                logging.info(f"Tool already has conn_id: {existing_conn_id}")
                                if existing_conn_id != st.session_state.conn_id:
                                    logging.warning(f"Replacing incorrect conn_id: {existing_conn_id} with {st.session_state.conn_id}")
                            
                            # Explicitly set the connection ID from session state
                            tool_input["conn_id"] = st.session_state.conn_id
                            
                            # Log the query execution
                            logging.info(f"Executing query with connection ID: {st.session_state.conn_id}")
                            if "query" in tool_input:
                                query_preview = tool_input["query"][:50] + "..." if len(tool_input["query"]) > 50 else tool_input["query"]
                                logging.info(f"Query: {query_preview}")
                            
                            # Execute the tool with the prepared input
                            result = await session.call_tool(
                                tool_use.name,
                                tool_input
                            )
                        else:
                            # For other tools, execute normally
                            result = await session.call_tool(
                                tool_use.name,
                                cast(dict, tool_use.input)
                            )

                        # Extract and format results
                        if hasattr(result, 'content') and result.content:
                            content = result.content[0]
                            if hasattr(content, 'text'):
                                result_text = content.text.strip()
                                
                                # Display the SQL query
                                with st.expander("📜 Executed SQL Query"):
                                    try:
                                        sql_display = tool_use.input.get("query") if isinstance(tool_use.input, dict) else str(tool_use.input)
                                        if sql_display:
                                            st.code(sql_display, language='sql')
                                        else:
                                            st.write("Raw tool input:", tool_use.input)
                                    except Exception as e:
                                        st.warning("Failed to retrieve SQL query.")
                                        st.write("Raw tool input:", tool_use.input)

                                # Try to parse and display results as a table if possible
                                try:
                                    query_results = []

                                    for item in result.content:
                                        if hasattr(item, 'text'):
                                            text = item.text.strip()

                                            # Try JSON array first
                                            try:
                                                parsed = json.loads(text)
                                                if isinstance(parsed, list):
                                                    query_results.extend(parsed)
                                                    continue
                                                elif isinstance(parsed, dict):
                                                    query_results.append(parsed)
                                                    continue
                                            except json.JSONDecodeError:
                                                pass

                                            # Try line-delimited JSON objects
                                            for line in text.splitlines():
                                                try:
                                                    parsed_line = json.loads(line.strip())
                                                    query_results.append(parsed_line)
                                                except json.JSONDecodeError:
                                                    continue  # ignore unparsable lines

                                    if query_results:
                                        df = pd.DataFrame(query_results)
                                        st.dataframe(df, use_container_width=True)
                                        st.write(f"Total rows: {len(query_results)}")
                                        result_text = f"Query returned {len(query_results)} rows.\n\n{df.to_string()}"
                                    else:
                                        st.info("Query executed successfully but returned no parsable rows.")
                                        result_text = "Query returned no parsable rows."
                                except Exception as e:
                                    st.error(f"Error processing results: {e}")
                                    st.write("Failed to process results. Raw result:", result_text)
                                    st.code(result_text)

                                tool_result = {
                                    "type": "tool_result",
                                    "tool_use_id": tool_use.id,
                                    "content": result_text
                                }
                                tool_results.append(tool_result)
                                st.session_state.last_query_result = result_text

                            else:
                                st.warning("Query executed but returned an unexpected format.")
                        else:
                            st.info("Query executed successfully. No data returned (common for INSERT, UPDATE, DELETE operations).")
                            # Create a default message for non-returning queries to avoid empty content
                            result_text = "Operation completed successfully. No data was returned (typical for INSERT, UPDATE, DELETE operations)."
                            tool_result = {
                                "type": "tool_result",
                                "tool_use_id": tool_use.id,
                                "content": result_text
                            }
                            tool_results.append(tool_result)
                            st.session_state.last_query_result = result_text

                    except Exception as tool_error:
                        st.error(f"Tool execution error: {tool_error}")
                        # Add the error to tool results so the AI can learn from it
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_use.id,
                            "content": f"Error: {str(tool_error)}"
                        })

                # Ensure we're not sending empty content to the API
                if tool_results:
                    messages.append({
                        "role": "user",
                        "content": tool_results
                    })
                else:
                    # Fallback for when no tool results were generated
                    messages.append({
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": tool_uses[0].id if tool_uses else "fallback_id",
                            "content": "The operation was processed but no specific result data was returned."
                        }]
                    })

            st.session_state.sql_finished = True
            
        except Exception as e:
            st.error(f"Query processing error: {e}")
            logging.error(f"Query processing error: {e}")



    async def connect_to_database(self, session):
        """Connect to the PostgreSQL database and store connection ID"""
        if not self.db_url:
            st.error("Database URL not provided. Please configure it in the sidebar.")
            return False
        
        try:
            # Use the connect tool to register the connection
            connect_result = await session.call_tool(
                "connect",
                {
                    "connection_string": self.db_url
                }
            )
            
            # Extract connection ID
            if hasattr(connect_result, 'content') and connect_result.content:
                content = connect_result.content[0]
                if hasattr(content, 'text'):
                    result_data = json.loads(content.text)
                    conn_id = result_data.get('conn_id')
                    st.session_state.conn_id = conn_id
                    
                    # Fetch schema information
                    with st.status("Fetching database schema..."):
                        st.session_state.schema_info = await self.fetch_schema_info(session, conn_id)
                    
                    return True
            
            st.error("Failed to connect to database: Invalid response from server")
            return False
            
        except Exception as e:
            st.error(f"Failed to connect to database: {e}")
            logging.error(f"Database connection error: {e}")
            return False

    async def run_async(self):
        model, max_tokens = self.render_sidebar()

        # Use SSE transport for PostgreSQL
        async with sse_client(url=self.pg_mcp_url) as streams:
            async with ClientSession(*streams) as session:
                await session.initialize()

                # Ensure connection ID is set
                if not st.session_state.conn_id:
                    if not await self.connect_to_database(session):
                        return
                
                # Only fetch schema if not already available
                if not st.session_state.schema_info:
                    # Fetch schema only if it's not in session_state
                    with st.status("Fetching database schema..."):
                        st.session_state.schema_info = await self.fetch_schema_info(session, st.session_state.conn_id)

                # Process the user query
                query = st.chat_input("Enter your natural language query...")

                if query:
                    st.session_state.sql_finished = False
                    await self.process_query(
                        session,
                        query,
                        model=model,
                        max_tokens=max_tokens
                    )

                if st.session_state.get("sql_finished"):
                    with st.expander("📊 AI-Generated Visualizations"):
                        await self.generate_visualizations(model)
                        st.session_state.sql_finished = False

    def run(self):
        self.render_header()
        asyncio.run(self.run_async())

def main():
    app = PostgreSQLAssistantApp()
    app.run()

if __name__ == "__main__":
    main()
