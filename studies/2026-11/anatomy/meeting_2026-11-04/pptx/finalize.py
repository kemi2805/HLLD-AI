"""Set the document title and replace the template's embedded thumbnail
(its 'Mon Nov 01' cover) with a render of this deck's cover."""
import datetime
import io
import sys

from PIL import Image
from pptx import Presentation

deck, cover_png, out = sys.argv[1:4]
prs = Presentation(deck)
cp = prs.core_properties
cp.title = "Meeting with Luciano, 4 Nov 2026: exact RMHD on the four-quadrant problem"
cp.modified = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None, microsecond=0)
cp.revision = (cp.revision or 0) + 1
buf = io.BytesIO()
Image.open(cover_png).convert("RGB").resize((256, 144), Image.LANCZOS).save(buf, "JPEG", quality=88)
for part in prs.part.package.iter_parts():
    if str(part.partname) == "/docProps/thumbnail.jpeg":
        part._blob = buf.getvalue()
        print("thumbnail replaced")
prs.save(out)
print("wrote", out)
