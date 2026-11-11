# Exports the Word paper to PDF with Microsoft Word, so the PDF matches Word's own layout.
#   powershell -ExecutionPolicy Bypass -File research_paper\scripts\export_pdf.ps1
param(
    [string]$Docx = (Join-Path $PSScriptRoot "..\hindi_english_code_switching_asr.docx"),
    [string]$Pdf = (Join-Path $PSScriptRoot "..\hindi_english_code_switching_asr.pdf")
)
$Docx = (Resolve-Path $Docx).Path
$Pdf = [System.IO.Path]::GetFullPath($Pdf)
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open($Docx, $false, $true)
    $doc.Repaginate()
    $pages = $doc.ComputeStatistics(2)
    $doc.SaveAs2($Pdf, 17)
    $doc.Close($false)
    Write-Output "exported $Pdf ($pages pages)"
}
finally {
    $word.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
}
