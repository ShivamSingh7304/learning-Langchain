#for suppressing the deprecation warning
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path='DOCUMENT_LOADERS/fifa_world_cup_2026_player_performance.csv')

data = loader.load()

print(data[100].page_content)