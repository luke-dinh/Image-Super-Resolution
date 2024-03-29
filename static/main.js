$(document).ready(function(){
    //Init
    $('.image-dection').hide();
    $('.loader').hide();
    $('#result').hide();

     // Upload Preview
    //  function readURL(input) {
    //     if (input.files && input.files[0]) {
    //         var reader = new FileReader();
    //         reader.onload = function (e) {
    //             $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
    //             $('#imagePreview').hide();
    //             $('#imagePreview').fadeIn(650);
    //         }
    //         reader.readAsDataURL(input.files[0]);
    //     }
    // }
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();

            reader.onload = function (e) {
                $('#blah')
                    .attr('src', e.target.result)
                    .width(100)
                    .height(100);
            };

            reader.readAsDataURL(input.files[0]);
        }
    }
    $("#imageUpload").change(function () {
        $('.image-section').show();
        $('#btn-predict').show();
        $('#result').text('');
        $('#result').hide();
        readURL(this);
    });

    // Predict
    $('#btn-predict').click(function () {
        var form_data = new FormData($('#upload-file')[0]);

        // Show loading animation
        $(this).hide();
        $('.loader').show();

        // Make prediction by calling api /predict
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            // success: function (data) {
            //     // Get and display the result
            //     $('.loader').hide();
            //     $('#result').fadeIn(600);
            //     $('#result').text(' Result:  ' + data);
            //     console.log('Success!');
            // },
            success: function show_image(data, width, height, alt){
                var img = document.createElement("img");
                img.data = data
                img.width = width;
                img.height = height;
                img.alt = alt;

                document.body.appendChild(img);
            }
        });
    });

});