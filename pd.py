from reportlab.lib.pagesizes import A5, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT

# Data for the timeline
data = [
    ["Year", "Event", "Key Developments"],
    ["1192", "Defeat of King Prithviraj Chauhan",
     "Prithviraj Chauhan was defeated by Turkic and Afghan forces.\n"
     "This led to the rise of the Delhi Sultanate."],
    ["1206", "Delhi Sultanate Established",
     "Delhi became a political centre.\n"
     "Five Turkic-Afghan dynasties ruled with instability and violent successions."],
    ["1326", "Mewar Kingdom Revived",
     "Mewar became a Rajput stronghold.\n"
     "It resisted the Delhi Sultanate and later the Mughals."],
    ["1336", "Vijayanagara Empire",
     "Harihara and Bukka founded Vijayanagara (Hampi).\n"
     "It grew into a strong southern empire."],
    ["1347", "Bahmani Sultanate",
     "Formed in the Deccan.\n"
     "It later split into five Sultanates, rivals of Vijayanagara."],
    ["1398", "Timur attacks Delhi",
     "Timur invaded and plundered Delhi.\n"
     "The city was ruined and the Sultanate weakened."],
    ["1498", "Portuguese Arrive",
     "Portuguese traders reached India.\n"
     "They sold horses and wrote travel accounts."],
    ["1526", "First Battle of Panipat",
     "Babur defeated Ibrahim Lodi with gunpowder.\n"
     "This started the Mughal Empire."],
    ["1556", "Second Battle of Panipat",
     "Akbar’s forces defeated Himu.\n"
     "Mughal rule in Delhi was secured."],
    ["1565", "Battle of Talikota",
     "Deccan Sultanates defeated Vijayanagara.\n"
     "The city was destroyed and empire collapsed."],
    ["1576", "Battle of Haldighati",
     "Maharana Pratap resisted Mughals.\n"
     "He escaped and continued guerrilla war."],
    ["1671", "Battle of Saraighat",
     "Ahom commander Lachit Borphukan defeated Mughals.\n"
     "This preserved Assam’s independence."],
    ["1699", "Formation of Khalsa",
     "Guru Gobind Singh created the Khalsa.\n"
     "It defended Sikhism against Mughal persecution."],
    ["1754", "Marathas control Delhi",
     "Marathas captured Delhi.\n"
     "This marked Mughal decline."],
    ["1799", "Sikh Empire Established",
     "Maharaja Ranjit Singh united Sikhs.\n"
     "A strong empire rose in Punjab."],
]

# Define the output file path
pdf_file = "Timeline_A5_Corrected.pdf"

# Set up the document template with landscape A5 pagesize and margins
doc = SimpleDocTemplate(pdf_file, pagesize=landscape(A5), rightMargin=20, leftMargin=20, topMargin=20, bottomMargin=20)

# --- FIX: Convert strings to Paragraph objects for automatic text wrapping ---
# Get default styles
styles = getSampleStyleSheet()
# Create a custom style for the body text to match the desired font size
body_style = ParagraphStyle(
    name='Body',
    parent=styles['BodyText'],
    fontSize=12,
    leading=14,  # Line spacing
    alignment=TA_LEFT
)

# Process the data list to wrap text content in Paragraph objects
processed_data = []
# Add the header row as is
processed_data.append(data[0])

# Loop through the data rows and convert text to Paragraphs
for row in data[1:]:
    year = row[0]
    event = Paragraph(row[1], body_style)
    # Replace newline characters with <br/> for line breaks inside a Paragraph
    developments_text = row[2].replace('\n', '<br/>')
    developments = Paragraph(developments_text, body_style)
    processed_data.append([year, event, developments])
# --- End of fix ---

# Create the table using the processed data and adjusted column widths
# Landscape A5 is 595.27 points wide. With 20pt margins, we have ~555 points.
table = Table(processed_data, colWidths=[50, 140, 345])

# Apply styles to the table
table.setStyle(TableStyle([
    # Header styles
    ("BACKGROUND", (0, 0), (-1, 0), colors.lightblue),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
    ("ALIGN", (0, 0), (-1, 0), "LEFT"),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, 0), 14),
    ("LINEBELOW", (0, 0), (-1, 0), 1, colors.black),

    # Body styles
    ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
    ("VALIGN", (0, 0), (-1, -1), "TOP"), # Vertically align all content to the top
    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
]))

# Build the PDF document
elements = [table]
doc.build(elements)

print(f"Successfully created PDF: {pdf_file}")