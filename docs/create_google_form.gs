/**
 * KNUST Student Cybersecurity Behaviour Survey — Google Form Generator
 * ---------------------------------------------------------------------
 * HOW TO USE:
 * 1. Go to https://script.google.com  ->  New project
 * 2. Delete any default code, paste ALL of this file
 * 3. Click Save (disk icon)
 * 4. Select the function "createKNUSTForm" from the dropdown at the top
 * 5. Click Run. Approve permissions when asked (first run only).
 * 6. Open the Execution log — it prints the live form URL and edit URL.
 *
 * The five-point behaviour items (Q6-Q21) all use the SAME scale:
 *   Never / Rarely / Sometimes / Often / Always
 * so the responses form one coherent trait (needed for the ML model).
 *
 * NOTE: This creates the STUDENT-FACING form. It deliberately does NOT
 * include the scoring guide or design notes — those stay in your Word doc.
 */

function createKNUSTForm() {
  var form = FormApp.create('KNUST Student Cybersecurity Behaviour Survey');
  form.setTitle('KNUST Student Cybersecurity Behaviour Survey');
  form.setDescription(
    'A study on student cybersecurity behaviour at KNUST. Participation is ' +
    'voluntary and anonymous. No personally identifiable information (name, ' +
    'student ID, email) is collected. You may withdraw at any time. The survey ' +
    'takes about 8-10 minutes. By selecting "I agree" below, you consent to participate.'
  );
  form.setProgressBar(true);
  form.setCollectEmail(false);   // keep anonymous
  form.setLimitOneResponsePerUser(false);

  var FREQ = ['Never', 'Rarely', 'Sometimes', 'Often', 'Always'];

  // ---------- CONSENT ----------
  form.addMultipleChoiceItem()
    .setTitle('Do you consent to participate in this study?')
    .setChoiceValues(['I agree to participate', 'I do not agree'])
    .setRequired(true);

  // ---------- PART A: BACKGROUND & EXPERIENCE (Features) ----------
  form.addSectionHeaderItem()
    .setTitle('Part A — Background and Experience')
    .setHelpText('A few background questions about you.');

  form.addMultipleChoiceItem()
    .setTitle('Q1. What is your current level of study?')
    .setChoiceValues(['Level 100','Level 200','Level 300','Level 400','Postgraduate (Masters / PhD)'])
    .setRequired(true);

  form.addMultipleChoiceItem()
    .setTitle('Q2. Which college/faculty are you in?')
    .setChoiceValues([
      'College of Engineering','College of Science',
      'College of Humanities and Social Sciences','College of Art and Built Environment',
      'College of Health Sciences','College of Agriculture and Natural Resources','Other'])
    .setRequired(true);

  form.addMultipleChoiceItem()
    .setTitle('Q3. How many years have you actively used the internet?')
    .setChoiceValues(['Less than 2 years','2 to 4 years','5 to 7 years','More than 7 years'])
    .setRequired(true);

  form.addMultipleChoiceItem()
    .setTitle('Q4. Have you ever received formal cybersecurity training (a course, workshop, or certified session)?')
    .setChoiceValues([
      'Yes, a full course or certification','Yes, a short workshop or session',
      'Only informal (videos, articles)','No formal or informal training'])
    .setRequired(true);

  form.addMultipleChoiceItem()
    .setTitle('Q5. Have you (or someone close to you) ever been a victim of an online scam, hacking, or fraud?')
    .setChoiceValues([
      'Yes, it happened to me directly','Yes, to someone close to me',
      'No, but I know of cases','No, never'])
    .setRequired(true);

  // ---------- PART B (Part 1): FEATURE ITEMS ----------
  form.addSectionHeaderItem()
    .setTitle('Part B — Security Practices (1 of 2)')
    .setHelpText('For each statement, choose how often it is true of you: Never, Rarely, Sometimes, Often, or Always.');

  var featureItems = [
    'Q6. I check that a website address starts with https and shows a padlock before entering my password.',
    'Q7. I verify the sender\'s email address before clicking links in emails.',
    'Q8. I use a different password for each of my important accounts.',
    'Q9. I keep the operating system and apps on my devices updated.',
    'Q10. I lock my screen or log out when I leave a shared or public computer.',
    'Q11. I avoid downloading apps or files from unofficial sources.',
    'Q12. I think carefully before sharing personal information online.',
    'Q13. I back up important files so I would not lose them to ransomware or device loss.'
  ];
  featureItems.forEach(function(t) {
    form.addMultipleChoiceItem().setTitle(t).setChoiceValues(FREQ).setRequired(true);
  });

  // ---------- PART B (Part 2): LABEL ITEMS ----------
  form.addSectionHeaderItem()
    .setTitle('Part B — Security Practices (2 of 2)')
    .setHelpText('A few more practices. Same scale: Never, Rarely, Sometimes, Often, or Always.');

  var labelItems = [
    'Q14. I use two-factor authentication (2FA) on my important accounts.',
    'Q15. I hesitate and verify before responding to urgent or alarming messages.',
    'Q16. I avoid using public Wi-Fi for sensitive activities, or I use a VPN when I do.',
    'Q17. I check app permissions before installing and remove apps I no longer trust.',
    'Q18. I report or delete suspicious emails and messages rather than acting on them.',
    'Q19. I use strong passwords or a password manager rather than simple, memorable ones.',
    'Q20. I review the privacy and security settings on my social media and email accounts.',
    'Q21. I avoid clicking links or attachments from senders I do not recognise.'
  ];
  labelItems.forEach(function(t) {
    form.addMultipleChoiceItem().setTitle(t).setChoiceValues(FREQ).setRequired(true);
  });

  // ---------- DONE ----------
  Logger.log('FORM CREATED SUCCESSFULLY');
  Logger.log('Live form (share this with students): ' + form.getPublishedUrl());
  Logger.log('Edit the form here: ' + form.getEditUrl());
  Logger.log('Responses will appear in the form\'s Responses tab. ' +
             'Click the Sheets icon there to send them to a Google Sheet.');
}
