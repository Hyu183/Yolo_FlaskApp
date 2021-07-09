$(document).ready(function () {
    $('#loader').hide();

    $('.loader-nav').hide();

    $('#btn-detect').click(function(e){
        $('#loader').show();
    });

    $('.load-model').click(function(e){
        $('.loader-nav').show();
    })
    
    
    $('input:submit').attr('disabled',true);
    $('input:file').change(
        function(){
            if ($(this).val()){
                $('input:submit').removeAttr('disabled'); 
            }
            else {
                $('input:submit').attr('disabled',true);
            }
        });
});
