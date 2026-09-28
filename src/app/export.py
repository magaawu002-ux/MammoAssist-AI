"""Anonymous, in-memory research exports; never include upload names or DICOM IDs."""

from io import BytesIO
from pathlib import Path
import textwrap

import numpy as np
from PIL import Image,ImageDraw,ImageFont

from src.app.core import TITLE,DISCLAIMER,SCORE_NOTE,ATTENTION_NOTE,DOMAIN_NOTE,THRESHOLD,TARGET


def report_lines(result,reference,timestamp,warnings):
    # Caller supplies a randomly generated session reference, not a user string.
    import re
    if not re.fullmatch(r'upload-[a-f0-9]{12}',reference): raise ValueError('Anonymous reference required')
    return [TITLE,DISCLAIMER,'UTC timestamp: '+timestamp,'Anonymous reference: '+reference,
            'Model: ResNet50 | Input: 224 x 224 | Preprocessing: canonical_v1 (no CLAHE)',
            f"Model output: malignant-pattern score = {result['score']:.6f}",
            f'Research decision threshold = {THRESHOLD}',result['decision'],SCORE_NOTE,
            'Grad-CAM target: '+TARGET,ATTENTION_NOTE,DOMAIN_NOTE,
            'Input-quality heuristics (not clinical certification):']+ (warnings or ['No heuristic warning; this does not certify clinical quality.']) +[
            'Review images for burned-in identifying text before sharing. Image pixels are not automatically de-identified.']


def png_report(result,canonical,reference,timestamp,warnings):
    lines=report_lines(result,reference,timestamp,warnings)
    # Matplotlib bundles this Unicode font on Windows too; no system-font guess.
    import matplotlib
    font_path=Path(matplotlib.get_data_path())/'fonts/ttf/DejaVuSans.ttf'
    font=ImageFont.truetype(str(font_path),19); titlefont=ImageFont.truetype(str(font_path),26)
    wrapped=[line for item in lines for line in textwrap.wrap(item,width=115)]
    canvas=Image.new('RGB',(1280,660+len(wrapped)*27),'#ffffff'); draw=ImageDraw.Draw(canvas)
    draw.text((30,20),'MammoAssist AI | Research summary',font=titlefont,fill='#152f46')
    panels=[np.repeat(canonical[...,None],3,axis=-1),np.repeat(result['heatmap'][...,None],3,axis=-1),result['overlay']]
    for x,array,label in zip((30,450,870),panels,('Canonical full mammogram','Grad-CAM attention (0–1)','Model-attention overlay')):
        picture=Image.fromarray(np.uint8(np.clip(array,0,1)*255)).resize((380,380),Image.Resampling.NEAREST)
        canvas.paste(picture,(x,95)); draw.text((x,65),label,font=font,fill='#152f46')
    y=500
    for line in wrapped:
        draw.text((30,y),line,font=font,fill='#182838');y+=27
    output=BytesIO();canvas.crop((0,0,1280,y+30)).save(output,format='PNG');return output.getvalue()


def pdf_report(result,canonical,reference,timestamp,warnings):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Image as ReportImage
    from xml.sax.saxutils import escape
    output=BytesIO(); styles=getSampleStyleSheet()
    doc=SimpleDocTemplate(output,pagesize=A4,title='MammoAssist AI research summary',author='MammoAssist AI',pageCompression=0)
    lines=report_lines(result,reference,timestamp,warnings)
    flow=[Paragraph(escape(lines[0]),styles['Title'])]
    for line in lines[1:]: flow.extend([Paragraph(escape(line),styles['BodyText']),Spacer(1,5)])
    picture=BytesIO();Image.fromarray(np.uint8(result['overlay']*255)).save(picture,format='PNG');picture.seek(0)
    flow += [Paragraph('Grad-CAM model-attention overlay (not a lesion boundary)',styles['Heading2']),ReportImage(picture,width=240,height=240)]
    doc.build(flow);return output.getvalue()
