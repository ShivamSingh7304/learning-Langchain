from langchain_text_splitters import CharacterTextSplitter

text ='''Football, also known as association football or soccer in some countries, is one of the most popular sports in the world. It is played and followed by millions of people across different continents. The sport is known for its simplicity, teamwork, strategy, physical fitness, and global popularity.
Introduction to Football
Football is a team sport played between two teams. Each team normally consists of eleven players, including one goalkeeper. The main objective of the game is to score goals by moving the ball into the opponent's goal.
Players generally use their feet to control, pass, and shoot the ball. Except for the goalkeeper within the permitted penalty area, players are not allowed to deliberately handle the ball with their hands or arms.
The team that scores more goals by the end of the match wins the game. If both teams score the same number of goals, the match may end in a draw. In knockout competitions, extra time and a penalty shootout may be used to determine the winner.
History of Football
Different forms of ball games have existed for thousands of years. Ancient civilizations in China, Greece, and Rome played games involving balls. However, modern football developed mainly in England during the nineteenth century.
Different schools and clubs initially followed their own rules. This created confusion when teams attempted to play against each other.
In 1863, the Football Association was formed in England. It helped establish a common set of rules for association football. These standardized rules separated association football from other sports such as rugby.
Football gradually spread throughout Europe and other parts of the world. International competitions and professional football clubs contributed to the rapid growth of the sport.'''

splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=5,
    separator=""
    
)
result = splitter.split_text(text)

print(result)