from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail


def send_email(api_key, email_to, s3_paths, display_user=False, error_log=None):
    '''
    Sends email to the specified email address


    :param email_to:
    :param s3_paths: dict of name to uploaded file. name will appear as link text in the email. specify None if we ran into error
    :return:
    '''

    if isinstance(s3_paths, dict):
        subject = 'Your results are ready!'
        body = '<p>Your following files will be securely stored:</p>'
        for name, path in s3_paths.items():
            body += "<p><a href='{}'>{}</a></p>".format(path, name)

    else:
        subject = 'We ran into an error generating your results'
        body = '<p>The following files ran into an error:</p>'
        # we are passed in a list of slide names
        for name in s3_paths:
            body += "<p>{}</p>".format(name)
        body += "<p>We apologize for this inconvenience. " \
                "<a href='www.pathologyreports.ai'>Please try again</a></p>"

    if error_log is not None:
        body += '<p>{}</p>'.format(str(error_log))

    if display_user:
        body += '<p>Submitted by: {}</p>'.format(email_to)

    message = Mail(
        from_email='brainwebsiteresults@brainii.com',
        to_emails=email_to,
        subject=subject,
        html_content=body
    )

    try:
        sg = SendGridAPIClient(api_key)
        response = sg.send(message)
        print(response.status_code)
        print(response.body)
        print(response.headers)
    except Exception as e:
        print(e.message)
