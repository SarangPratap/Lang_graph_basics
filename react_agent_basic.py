from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import create_agent
from dotenv import load_dotenv
import requests
import datetime

load_dotenv()





llm = ChatOpenAI(model="gpt-5-nano")
result = llm.invoke("What is the capital of France?")
print(result.content)

# result = llm.invoke("What is the weather like in Germany today?")
# print(result.content)

def location_to_coordinates(location: str) -> str:
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={location}&count=10&language=en&format=json"
    response = requests.get(url)
    data = response.json()
    
    if "results" in data:
        results = data["results"][0]
        lat= results.get("latitude")
        lon = results.get("longitude")
        return lat,lon
    else:
        return f"Could not find coordinates for location: {location}"
    
    # we will call the api here to get the coordinates of the location
    
@tool("get_weather") 
def get_weather(location: str) -> str:
    """Get the weather for a specific location. Use this to answer weather questions."""
    try:
        latitude, longitude = location_to_coordinates(location)
    except Exception as e:
        return f"Error occurred while fetching weather for {location}: {e}"

    url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&daily=temperature_2m_max&timezone=auto"
    response = requests.get(url, timeout= 28)
    response.raise_for_status()  # Check if the request was successful
    data= response.json()
    
    temps = data.get("daily", {}).get("temperature_2m_max")
    
    if not temps:
        return f"I found {location}, but couldn't read temperature data for {datetime.date.today().isoformat()}."

    return f"Weather for {location} on {datetime.date.today().isoformat()}: max temperature {temps[0]}°C."

agent = create_agent(llm, tools=[get_weather])

result = agent.invoke({"messages": [("human", "What is the weather like in New delhi today?")]})  # should be like dict with messages as list of tuples

print(result["messages"][-1].content)
