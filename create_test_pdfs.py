from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from pathlib import Path


DOCUMENTS_FOLDER = Path("documents")
DOCUMENTS_FOLDER.mkdir(exist_ok=True)


def create_pdf(filename, title, content):

    filepath = DOCUMENTS_FOLDER / filename

    pdf = canvas.Canvas(str(filepath), pagesize=A4)

    width, height = A4

    y = height - 60

    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(50, y, title)

    y -= 40

    pdf.setFont("Helvetica", 11)

    for line in content.split("\n"):

        if y < 60:
            pdf.showPage()
            pdf.setFont("Helvetica", 11)
            y = height - 60

        pdf.drawString(50, y, line)

        y -= 18

    pdf.save()

    print(f"Created: {filepath}")


inspection_text = """
USED VEHICLE INSPECTION REPORT

Vehicle:
2019 Example Sedan

Mileage:
68,500 km

Engine:
Engine starts normally.
No visible oil leakage observed.

Transmission:
Transmission shifts normally during inspection.

Brakes:
Front brake pads approximately 40% remaining.

Tyres:
Front tyres approximately 30% remaining.
Rear tyres approximately 50% remaining.

Body:
Minor scratches on rear bumper.

Engine warning light:
No warning light observed during inspection.

Recommendation:
Vehicle should undergo a professional diagnostic scan before purchase.
"""


service_text = """
VEHICLE SERVICE HISTORY

Vehicle:
2019 Example Sedan

15,000 km:
Regular service completed.

30,000 km:
Regular service completed.

45,000 km:
Brake pads replaced.

60,000 km:
Transmission fluid replaced.

65,000 km:
Engine sensor replaced.

68,000 km:
Regular service completed.
"""


insurance_text = """
INSURANCE HISTORY

Vehicle:
2019 Example Sedan

2021:
Insurance claim recorded for rear bumper damage.

2022:
No claim recorded.

2023:
No claim recorded.

2024:
No claim recorded.

2025:
No claim recorded.
"""


create_pdf(
    "vehicle_inspection.pdf",
    "Vehicle Inspection Report",
    inspection_text
)

create_pdf(
    "service_history.pdf",
    "Vehicle Service History",
    service_text
)

create_pdf(
    "insurance_history.pdf",
    "Insurance History",
    insurance_text
)


print("\nAll test PDFs created successfully!")